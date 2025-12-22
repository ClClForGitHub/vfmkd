"""
SAM2 PromptEncoder + MaskDecoder 适配头（最小可用版）。

输入：来自可替换的 backbone 的三层特征 (stride 8/16/32)。
适配：将各层用 1x1 卷积对齐到 256 通道，并重采样到 64x64 后融合为 SAM2 所需的 image_embeddings。
输出：调用 SAM2 的 PromptEncoder / MaskDecoder，根据点/框/掩码提示输出低分辨率掩码与 IoU 预测。
"""

from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..heads.base_head import BaseHead

# 引入 SAM2 的 PromptEncoder / MaskDecoder / TwoWayTransformer
# 直接导入SAM2组件，避免路径问题
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "vfmkd" / "sam2"))

from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import LayerNorm2d


class SAM2SegAdapterHead(BaseHead):
    """
    将自有 backbone 的多尺度特征适配到 SAM2 的提示编码器与掩码解码器，实现点/框→掩码的最小闭环。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 关键超参（与本次最小验证保持一致）
        self.image_size: int = int(config.get("image_size", 1024))  # 统一 1024
        self.hidden_dim: int = int(config.get("hidden_dim", 256))  # SAM/SAM2 头的 embed_dim
        self.backbone_strides: List[int] = config.get("backbone_strides", [8, 16, 32])
        self.use_high_res_features: bool = bool(config.get("use_high_res_features", False))
        self.num_multimask_outputs: int = int(config.get("num_multimask_outputs", 3))

        # 目标 image embedding 的网格尺寸（SAM/SAM2 默认 stride=16 → 64x64 @ 1024）
        self.target_stride: int = int(config.get("target_stride", 16))
        self.target_hw: int = self.image_size // self.target_stride

        # 支持两种模式：单层 s16（默认）或多层融合
        self.pyramid_mode: str = str(config.get("pyramid_mode", "s16"))  # "s16" | "p3p4p5"

        # 通道对齐 1x1 卷积
        in_channels: List[int] = config.get("in_channels", [256, 512, 1024])  # 与实际 backbone 对齐
        assert len(in_channels) == 3, "in_channels 需提供三层的通道数 [s8, s16, s32]"

        self.proj_s8 = nn.Conv2d(in_channels[0], self.hidden_dim, kernel_size=1, bias=False)
        self.proj_s16 = nn.Conv2d(in_channels[1], self.hidden_dim, kernel_size=1, bias=False)
        self.proj_s32 = nn.Conv2d(in_channels[2], self.hidden_dim, kernel_size=1, bias=False)

        # 最小对齐：LayerNorm2d（与SAM分布更一致），避免过强的L2归一导致饱和
        self.ln = LayerNorm2d(self.hidden_dim)

        # 构建 SAM 风格 PromptEncoder / MaskDecoder（与 SAM2Base._build_sam_heads 对齐）
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.hidden_dim,
            image_embedding_size=(self.target_hw, self.target_hw),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=self.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features,
            iou_prediction_use_sigmoid=True,
        )

    def _build_image_embeddings(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        将 [s8, s16, s32] 三层特征对齐到 (B, 256, 64, 64) 并融合。
        """
        assert len(features) >= 3, "需要提供至少三层特征 [s8, s16, s32]"
        feat_s8, feat_s16, feat_s32 = features[:3]

        target_hw = (self.target_hw, self.target_hw)

        if self.pyramid_mode == "s16":
            # 仅使用 stride=16 特征，最贴近 SAM/SAM2 的主干输出标尺
            x16 = self.proj_s16(feat_s16)
            x16 = self.ln(x16)
            x16 = F.interpolate(x16, size=target_hw, mode="bilinear", align_corners=False)
            return x16

        # 多层融合（可选）
        x8 = self.proj_s8(feat_s8)
        x16 = self.proj_s16(feat_s16)
        x32 = self.proj_s32(feat_s32)
        x8 = F.interpolate(x8, size=target_hw, mode="bilinear", align_corners=False)
        x16 = F.interpolate(x16, size=target_hw, mode="bilinear", align_corners=False)
        x32 = F.interpolate(x32, size=target_hw, mode="bilinear", align_corners=False)
        x = (x8 + x16 + x32) / 3.0
        x = self.ln(x)
        return x

    @torch.no_grad()
    def get_dense_pe(self) -> torch.Tensor:
        return self.prompt_encoder.get_dense_pe()

    def forward(
        self,
        features: List[torch.Tensor],
        *,
        point_coords: Optional[torch.Tensor] = None,  # [B, N, 2]，在 1024 输入坐标系
        point_labels: Optional[torch.Tensor] = None,  # [B, N]，1/0/-1
        boxes: Optional[torch.Tensor] = None,         # [B, 4]，在 1024 输入坐标系
        mask_inputs: Optional[torch.Tensor] = None,   # [B, 1, H, W] 原图尺度掩码（可选）
        num_multimask_outputs: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        最小推理闭环：features + prompts → 低分辨率掩码与 IoU 预测。
        注意：返回的 low_res_logits 为 256x256，适合进一步上采样到原图。
        """
        image_embeddings = self._build_image_embeddings(features)  # (B, 256, 64, 64)

        # Prompt 编码（坐标已在 1024 坐标系，无需额外归一化，这由 PromptEncoder 处理）
        points = None
        if point_coords is not None and point_labels is not None:
            # PromptEncoder 期望 labels 为 int32
            if point_labels.dtype != torch.int32:
                point_labels = point_labels.to(torch.int32)
            points = (point_coords, point_labels)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_inputs,
        )

        # 使用基础模式，不传递high_res_features参数
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=bool(num_multimask_outputs or self.num_multimask_outputs),
            repeat_image=False,
        )

        return {
            "low_res_logits": low_res_masks,  # (B, C, 256, 256)
            "iou_predictions": iou_predictions,
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        最小验证用：无训练，返回空损失字典。
        如需训练，可在此对 low_res_logits 与 GT 掩码计算 BCE/Dice 等。
        """
        return {}



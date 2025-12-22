"""
SAM2分割头部管理器（单例模式，离线初始化）

功能：
1. 离线初始化：使用build_sam2加载完整SAM2模型，提取prompt_encoder和mask_decoder
2. 复用机制：首次初始化后，后续直接使用缓存的组件
3. 掩码生成：接受对齐后的学生特征(64×64×256)和框提示，生成分割掩码
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

# 复用同一目录下的builder
from .builder import build_sam2


class SAM2SegHead:
    """
    SAM2分割头部管理器（单例模式，离线初始化）
    
    使用方式：
        # 初始化（首次调用，仅执行一次）
        head = SAM2SegHead.get_instance(device='cuda')
        
        # 生成掩码（可重复调用）
        result = head.generate_mask(
            aligned_features=student_features,  # (B, 256, 64, 64)
            boxes=box_tensor,                   # (B, 4) [x0, y0, x1, y1]
            multimask_output=True
        )
    """
    
    _instance: Optional['SAM2SegHead'] = None
    _initialized = False
    
    def __init__(self, device: torch.device):
        """私有构造函数，通过get_instance()调用"""
        if SAM2SegHead._initialized:
            raise RuntimeError("SAM2SegHead已初始化，请使用get_instance()获取实例")
        
        self.device = device
        self.prompt_encoder = None
        self.mask_decoder = None
        
        # 加载SAM2模型并提取组件（复用builder.py的build_sam2）
        self._load_sam2_heads()
        
        SAM2SegHead._initialized = True
    
    @classmethod
    def get_instance(cls, device: str = 'cuda') -> 'SAM2SegHead':
        """获取单例实例（懒加载模式）"""
        if cls._instance is None:
            device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
            cls._instance = cls(device_obj)
            print(f"[SAM2Head] 初始化完成，设备: {device_obj}")
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None
        cls._initialized = False
    
    def _load_sam2_heads(self):
        """加载SAM2模型并提取prompt_encoder和mask_decoder"""
        print(f"[SAM2Head] 加载SAM2完整模型...")
        
        # 复用builder.py的build_sam2函数
        sam2_model = build_sam2(device=self.device)
        
        # 提取组件（这些是引用，权重已通过build_sam2加载）
        if not hasattr(sam2_model, 'sam_prompt_encoder'):
            raise AttributeError("SAM2模型缺少sam_prompt_encoder属性")
        if not hasattr(sam2_model, 'sam_mask_decoder'):
            raise AttributeError("SAM2模型缺少sam_mask_decoder属性")
        
        self.prompt_encoder = sam2_model.sam_prompt_encoder
        self.mask_decoder = sam2_model.sam_mask_decoder
        
        # 确保在正确设备上并设置为eval模式
        self.prompt_encoder = self.prompt_encoder.to(self.device).eval()
        self.mask_decoder = self.mask_decoder.to(self.device).eval()
        
        # build_sam2已经设置了use_high_res_features = False，这里确认一下
        self.mask_decoder.use_high_res_features = False
        
        print(f"[SAM2Head] Prompt Encoder和Mask Decoder提取完成")
        print(f"[SAM2Head] 设备: {self.device}, High-res features: {self.mask_decoder.use_high_res_features}")
    
    @torch.no_grad()
    def generate_mask(
        self,
        aligned_features: torch.Tensor,  # (B, 256, 64, 64) - 已对齐的学生特征
        boxes: Optional[torch.Tensor] = None,  # (B, 4) 或 (N, 4) [x0, y0, x1, y1] 或 None
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ((B, N, 2), (B, N)) 或 None
        masks: Optional[torch.Tensor] = None,  # (B, 1, H, W) 或 None
        multimask_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        生成分割掩码
        
        Args:
            aligned_features: 已对齐的特征，形状(B, 256, 64, 64)
            boxes: 框提示，形状(B, 4)或(N, 4)，格式[x0, y0, x1, y1]，坐标在1024×1024坐标系
            points: 点提示元组(points_coords, point_labels)，或None
            masks: 掩码提示，形状(B, 1, H, W)，或None
            multimask_output: 是否输出多个掩码候选
        
        Returns:
            dict包含:
                - 'masks': (B, M, 256, 256) 低分辨率掩码logits，M=3 if multimask_output else 1
                - 'iou_predictions': (B, M) IOU预测值
                - 'best_mask': (B, 1, 256, 256) 最佳掩码（IOU最高）
                - 'best_iou': (B, 1) 最佳IOU值
                - 'best_idx': (B,) 最佳掩码索引
        """
        # 确保特征在正确设备上
        aligned_features = aligned_features.to(self.device)
        
        # 准备提示编码
        if boxes is not None:
            boxes = boxes.to(self.device)
            # 如果boxes是(N, 4)，需要确保与batch size匹配
            if boxes.dim() == 2 and boxes.shape[0] != aligned_features.shape[0]:
                # 如果只有一个框但有多batch，则复制
                if boxes.shape[0] == 1:
                    boxes = boxes.repeat(aligned_features.shape[0], 1)
        
        if points is not None:
            points_coords, point_labels = points
            points_coords = points_coords.to(self.device)
            point_labels = point_labels.to(self.device).to(torch.int32)  # prompt_encoder要求int32
            points_tuple = (points_coords, point_labels)
        else:
            points_tuple = None
        
        if masks is not None:
            masks = masks.to(self.device)
        
        # 提示编码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points_tuple,
            boxes=boxes,
            masks=masks,
        )
        
        # 获取位置编码
        image_pe = self.prompt_encoder.get_dense_pe()
        
        # 生成掩码
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=aligned_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=None,
        )
        
        # 选择最佳掩码（IOU最高）- 与参考代码完全一致
        # 参考：simple_comparison.py:270, final_clear_visual_from_scratch.py:112, diagnose_mask.py:102
        if multimask_output and low_res_masks.shape[1] > 1:
            # 直接对第一个batch取argmax，然后使用切片选择（与参考代码一致）
            best_idx = iou_predictions[0].argmax().item()  # 单个标量
            best_mask = low_res_masks[:, best_idx:best_idx+1]  # 使用切片，保持维度
            best_iou = iou_predictions[:, best_idx:best_idx+1]
            best_idx_tensor = torch.tensor([best_idx], dtype=torch.long, device=self.device)
        else:
            best_idx_tensor = torch.zeros(low_res_masks.shape[0], dtype=torch.long, device=self.device)
            best_mask = low_res_masks[:, 0:1]
            best_iou = iou_predictions[:, 0:1]
        
        return {
            'masks': low_res_masks,  # (B, M, 256, 256)
            'iou_predictions': iou_predictions,  # (B, M)
            'best_mask': best_mask,  # (B, 1, 256, 256)
            'best_iou': best_iou,  # (B, 1)
            'best_idx': best_idx_tensor,  # (B,)
        }
    
    @torch.no_grad()
    def upsample_mask(
        self,
        mask_logits: torch.Tensor,  # (B, 1, 256, 256) 或 (B, M, 256, 256)
        target_size: Tuple[int, int] = (1024, 1024),
        mode: str = 'bilinear',
        threshold: float = 0.0,
        return_binary: bool = False,
    ) -> torch.Tensor:
        """
        上采样掩码到目标尺寸
        
        Args:
            mask_logits: 低分辨率掩码logits
            target_size: 目标尺寸，默认(1024, 1024)
            mode: 插值模式，'bilinear'或'nearest'
            threshold: 二值化阈值（logits空间），默认0.0
            return_binary: 是否返回二值掩码
        
        Returns:
            上采样后的掩码，形状(B, C, H, W)
        """
        mask_logits = mask_logits.to(self.device)
        
        # 上采样
        upsampled = F.interpolate(
            mask_logits,
            size=target_size,
            mode=mode,
            align_corners=False if mode == 'bilinear' else None,
        )
        
        if return_binary:
            # 直接对logits进行二值化（logits > threshold），与参考代码一致
            # 注意：SAM2的mask_decoder输出的是logits，不是概率
            # 参考代码使用 mask > 0 进行二值化（threshold=0）
            binary = (upsampled > threshold).float()
            return binary
        else:
            return upsampled


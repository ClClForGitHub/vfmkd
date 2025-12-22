#!/usr/bin/env python3
"""
学生Backbone与Adapter加载 + 学生P4(64×64×256)特征提取封装。

要求：
- 权重文件若包含 {'backbone': ..., 'adapter': ...} 则分别加载；若仅包含 'adapter' 亦可（backbone随机初始化）。
- 学生预处理遵循YOLO路径：Resize(1024,1024) + ToTensor(/255.0)，不做ImageNet标准化。
- 返回的学生P4特征统一为 (1, 256, 64, 64)。
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import sys

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vfmkd.distillation.adapters import SimpleAdapterStatic


def load_student_backbone_and_adapter(
    weights_path: Union[str, Path],
    device: str = 'cuda:6',
    model_size: str = 's',
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    加载学生backbone与适配器：
    - 首选SimpleAdapter（训练脚本保存的feature_adapter）
    - 兼容旧格式的Sam2ImageAdapter（键名adapter）
    """
    # --- 延迟导入 ---
    from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
    from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter
    # --- 延迟导入结束 ---

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    cfg = {
        'model_size': model_size,
        'pretrained': False,
        'freeze_backbone': False,
    }
    backbone = YOLOv8Backbone(cfg).to(device_obj).eval()
    in_ch_s16 = backbone.get_feature_dims()[2] if len(backbone.get_feature_dims()) >= 3 else backbone.get_feature_dims()[-1]
    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)

    # 强制要求：必须存在并成功加载backbone与adapter权重
    if not isinstance(ckpt, dict):
        raise RuntimeError('权重文件格式不支持，需为dict')

    if 'backbone' not in ckpt or not isinstance(ckpt['backbone'], dict):
        raise RuntimeError('缺少backbone权重(backbone state_dict)。为避免随机初始化，已强制报错。')
    backbone.load_state_dict(ckpt['backbone'], strict=False)

    # 先尝试SimpleAdapter（feature_adapter，静态1x1）
    if 'feature_adapter' in ckpt and isinstance(ckpt['feature_adapter'], dict):
        adapter = SimpleAdapterStatic(in_channels=in_ch_s16, out_channels=256).to(device_obj).eval()
        adapter.load_state_dict(ckpt['feature_adapter'], strict=False)
        setattr(adapter, '_adapter_type', 'simple_static')
    elif 'adapter' in ckpt and isinstance(ckpt['adapter'], dict):
        # 兼容旧格式：使用Sam2ImageAdapter
        adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device_obj).eval()
        adapter.load_state_dict(ckpt['adapter'], strict=False)
        setattr(adapter, '_adapter_type', 'sam2_image')
    else:
        raise RuntimeError('缺少feature_adapter或adapter权重。适配器为必要组件，已强制报错。')

    return backbone, adapter


@torch.no_grad()
def extract_student_p4_from_image(
    image_path: Union[str, Path],
    backbone: torch.nn.Module,
    device: str = 'cuda:6',
) -> Tuple[torch.Tensor, list[torch.Tensor]]:
    """
    从单张图片提取学生backbone特征，返回：
    - S16特征张量 (B, C, 64, 64)
    - [S8, S16, S32] 列表（供Sam2ImageAdapter兼容使用）
    预处理：Resize(1024,1024)+ToTensor(/255.0)。
    """
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    img = Image.open(str(image_path)).convert('RGB').resize((1024, 1024))
    x = TF.to_tensor(img).unsqueeze(0).to(device_obj)  # (1,3,1024,1024), [0,1]

    feats = backbone(x)  # 期望: [S4,S8,S16,S32]
    if not isinstance(feats, (list, tuple)) or len(feats) < 3:
        raise RuntimeError('YOLOv8Backbone返回的特征数量不足，期望至少包含S16层')

    feat_s16 = feats[2]
    # 提取 [S8, S16, S32] 以便兼容Sam2ImageAdapter
    feats_for_adapter = feats[1:4] if len(feats) >= 4 else feats[1:]
    return feat_s16, feats_for_adapter


@torch.no_grad()
def align_student_features(
    student_s16: torch.Tensor,
    feats_for_adapter: list[torch.Tensor],
    teacher_features: torch.Tensor,
    adapter: torch.nn.Module,
) -> torch.Tensor:
    """
    使用加载的适配器将学生S16特征对齐到SAM2特征空间。
    根据适配器类型自动选择调用方式。
    """
    if hasattr(adapter, '_adapter_type') and adapter._adapter_type in ('simple', 'simple_static'):
        # 新版静态适配器仅接受学生特征；兼容旧标记时也优先用此路径
        try:
            aligned = adapter(student_s16)
        except TypeError:
        aligned = adapter(student_s16, teacher_features)
    elif hasattr(adapter, '_adapter_type') and adapter._adapter_type == 'sam2_image':
        aligned = adapter(feats_for_adapter)
    else:
        # 尝试推断
        if isinstance(adapter, SimpleAdapter):
            aligned = adapter(student_s16, teacher_features)
        else:  # 默认按照Sam2ImageAdapter处理
            aligned = adapter(feats_for_adapter)

    if aligned.shape[-2:] != (64, 64):
        aligned = F.interpolate(aligned, size=(64, 64), mode='bilinear', align_corners=False)
    return aligned


__all__ = [
    'load_student_backbone_and_adapter',
    'extract_student_p4_from_image',
    'align_student_features',
]



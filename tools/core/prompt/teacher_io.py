#!/usr/bin/env python3
"""
教师特征IO：
- NPZ教师特征加载与对齐到(1,256,64,64)
- 实时教师P4提取（基于SAM2模型 forward_image 或 image_encoder 输出）
"""

from pathlib import Path
from typing import Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_teacher_npz(npz_path: Path, device: str = 'cuda:6') -> torch.Tensor:
    """
    加载教师NPZ特征，优先按键 ['P4_S16','IMAGE_EMB_S16','image_embedding','image_emb_s16']，
    统一输出 float32 (1,256,64,64)。若通道或空间不匹配则插值/裁切到匹配形状（通道仅在=256时直通）。
    """
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    npz = np.load(str(npz_path), allow_pickle=True)
    key = None
    for k in ['P4_S16', 'IMAGE_EMB_S16', 'image_embedding', 'image_emb_s16']:
        if k in npz.files:
            key = k
            break
    if key is None:
        raise KeyError('NPZ缺少P4键：期望 P4_S16/IMAGE_EMB_S16/image_embedding/image_emb_s16')

    arr = npz[key]
    ten = torch.tensor(arr, dtype=torch.float32)
    if ten.dim() == 3:
        ten = ten.unsqueeze(0)
    if ten.shape[1] != 256:
        # 若通道不为256，尝试通过1x1映射（此处保守处理：取前/后截断或均值压缩到256）
        # 简化：超过256取前256，少于256用重复或pad到256
        c = ten.shape[1]
        if c > 256:
            ten = ten[:, :256]
        else:
            pad = 256 - c
            ten = torch.cat([ten, ten[:, :pad]], dim=1)
    if ten.shape[-2:] != (64, 64):
        ten = F.interpolate(ten, size=(64, 64), mode='bilinear', align_corners=False)
    return ten.to(device_obj)


@torch.no_grad()
def extract_realtime_teacher_p4(
    sam2_model,
    image_tensor_1024: torch.Tensor,
) -> torch.Tensor:
    """
    从SAM2模型实时提取P4/vision_features对应的(1,256,64,64)。
    要求 image_tensor_1024 已按 SAM2Transforms 预处理且尺寸为(1,3,1024,1024)。
    """
    # 优先使用 forward_image 的backbone_fpn
    if hasattr(sam2_model, 'forward_image'):
        out = sam2_model.forward_image(image_tensor_1024)
        if isinstance(out, dict):
            if 'backbone_fpn' in out and len(out['backbone_fpn']) >= 3:
                feat = out['backbone_fpn'][2]
                if feat.shape[-2:] != (64, 64):
                    feat = F.interpolate(feat, size=(64, 64), mode='bilinear', align_corners=False)
                return feat
    # 回退到 image_encoder 的 vision_features
    if hasattr(sam2_model, 'image_encoder'):
        bko = sam2_model.image_encoder(image_tensor_1024)
        if isinstance(bko, dict) and 'vision_features' in bko:
            feat = bko['vision_features']  # (1,256,64,64)
            if feat.shape[-2:] != (64, 64):
                feat = F.interpolate(feat, size=(64, 64), mode='bilinear', align_corners=False)
            return feat
    raise RuntimeError('无法从SAM2提取(1,256,64,64)教师特征')


__all__ = ['load_teacher_npz', 'extract_realtime_teacher_p4']



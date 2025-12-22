#!/usr/bin/env python3
"""
特征/掩码评估指标：
- 余弦相似度 / L2 / MAE（自动尺寸/通道对齐）
- 掩码IOU、像素个数、像素占比
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def _align_for_similarity(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 统一为 (B, C, H, W)
    if a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    # 空间对齐
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
    # 通道对齐（简化：截断/重复到min(Ca, Cb) 或扩展到较大者）
    ca, cb = a.shape[1], b.shape[1]
    c = min(ca, cb)
    a = a[:, :c]
    b = b[:, :c]
    return a.contiguous(), b.contiguous()


@torch.no_grad()
def compute_similarity(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a, b = _align_for_similarity(a, b)
    # 展平到 (N, D)
    A = a.reshape(a.shape[0], -1).float()
    B = b.reshape(b.shape[0], -1).float()
    # 逐样本余弦
    cos = F.cosine_similarity(A, B, dim=1).mean().item()
    l2 = torch.sqrt(((A - B) ** 2).mean(dim=1)).mean().item()
    mae = (A - B).abs().mean(dim=1).mean().item()
    return {
        'cosine': float(cos),
        'l2': float(l2),
        'mae': float(mae),
    }


@torch.no_grad()
def mask_stats(mask_binary: torch.Tensor) -> Dict[str, float]:
    # 期望 (B,1,H,W) 且为 {0,1}
    if mask_binary.dtype != torch.float32:
        mask_binary = mask_binary.float()
    pixels = mask_binary.sum().item()
    total = mask_binary.numel()
    ratio = pixels / total if total > 0 else 0.0
    return {
        'pixels': float(pixels),
        'ratio': float(ratio),
    }


@torch.no_grad()
def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        # 简化空间对齐
        b = F.interpolate(b.float(), size=a.shape[-2:], mode='nearest')
    a = a.float()
    b = b.float()
    inter = (a * b).sum().item()
    union = (a + b - a * b).sum().item()
    return float(inter / union) if union > 0 else 0.0


__all__ = ['compute_similarity', 'mask_stats', 'mask_iou']



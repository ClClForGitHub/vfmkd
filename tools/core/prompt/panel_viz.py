#!/usr/bin/env python3
"""
Two-row panel visualization used in Experiment A.
- Column 1: original image (shown on both rows)
- Row 1: mean-channel heatmaps for student / NPZ teacher / realtime teacher features
- Row 2: corresponding masks with pixel statistics
- Rightmost column: numeric summary (feature similarity & mask IOU)
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch


def _to_numpy_image(img_tensor: torch.Tensor) -> np.ndarray:
    # img_tensor: (3,H,W) or (1,3,H,W)
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))  # HWC
    return img


def _mean_map(feat: torch.Tensor) -> np.ndarray:
    # feat: (1,C,H,W) or (C,H,W)
    if feat.dim() == 4:
        feat = feat[0]
    mean_map = feat.mean(dim=0).detach().cpu().numpy()
    return mean_map


def draw_panel(
    image_tensor_0_1: torch.Tensor,  # (1,3,1024,1024) or (3,1024,1024)
    feat_student: torch.Tensor,       # (1,256,64,64)
    feat_npz: torch.Tensor,           # (1,256,64,64)
    feat_teacher: torch.Tensor,       # (1,256,64,64)
    mask_student: torch.Tensor,       # dict或张量，若为dict读取 ['prob','binary']
    mask_npz: torch.Tensor,
    mask_teacher: torch.Tensor,
    sim_metrics: Dict[str, float],    # {'cosine_*','mae_*','l2_*'} 或压缩成三对
    mask_ious: Dict[str, float],      # 三对掩码IOU
    output_path: Path,
) -> Path:
    """
    绘制两行七图+统计板并保存。
    sim_metrics 建议包含：
      - 'cos_student_npz', 'cos_student_teacher', 'cos_npz_teacher'
      - 可选 'mae_*', 'l2_*'
    mask_ious 建议包含：'iou_student_npz','iou_student_teacher','iou_npz_teacher'
    """
    # 解析图像
    img_np = _to_numpy_image(image_tensor_0_1)

    # 解析掩码概率/二值
    def _get_prob_and_bin(m):
        if isinstance(m, dict):
            prob = m.get('prob')
            binary = m.get('binary')
            if prob is None and 'logits' in m:
                prob = torch.sigmoid(m['logits'])
        else:
            prob = m
            binary = (m > 0.5).float()
        return prob, binary

    prob_s, bin_s = _get_prob_and_bin(mask_student)
    prob_n, bin_n = _get_prob_and_bin(mask_npz)
    prob_t, bin_t = _get_prob_and_bin(mask_teacher)

    # 统计像素
    def _pix_stat(b):
        b = b.float()
        pixels = b.sum().item()
        total = b.numel()
        ratio = pixels / total if total > 0 else 0.0
        return pixels, ratio

    pix_s, rat_s = _pix_stat(bin_s)
    pix_n, rat_n = _pix_stat(bin_n)
    pix_t, rat_t = _pix_stat(bin_t)

    # 创建画布：2行8列（7图+统计板）
    fig = plt.figure(figsize=(4*8, 4*2))
    plt.suptitle('Panel-A: Student vs Teachers — Features & Masks', fontsize=14, fontweight='bold')

    # 原图（两行左侧共两格）
    ax_img = plt.subplot(2, 8, 1)
    ax_img.imshow(img_np)
    ax_img.set_title('Original Image', fontsize=10)
    ax_img.axis('off')
    ax_img2 = plt.subplot(2, 8, 1 + 8)
    ax_img2.imshow(img_np)
    ax_img2.axis('off')

    # 第一行：特征 mean-channel
    feats = [feat_student, feat_npz, feat_teacher]
    feat_titles = ['Student P4', 'Teacher (NPZ) P4', 'Teacher (Realtime) P4']
    for i, (f, title) in enumerate(zip(feats, feat_titles)):
        ax = plt.subplot(2, 8, 2 + i)
        m = _mean_map(f)
        im = ax.imshow(m, cmap='viridis')
        ax.set_title(f'{title}\nmean={m.mean():.4f}, std={m.std():.4f}', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 第二行：掩码（概率热力或二值）
    masks = [
        (prob_s, bin_s, 'Student Mask'),
        (prob_n, bin_n, 'Teacher (NPZ) Mask'),
        (prob_t, bin_t, 'Teacher (Realtime) Mask'),
    ]
    for i, (p, b, title) in enumerate(masks):
        ax = plt.subplot(2, 8, 2 + i + 8)
        pmap = p[0, 0].detach().cpu().numpy() if p.dim() == 4 else p.detach().cpu().numpy()
        im = ax.imshow(pmap, cmap='gray', vmin=0, vmax=1)
        pix = b.sum().item()
        tot = b.numel()
        ratio = pix / tot if tot > 0 else 0.0
        ax.set_title(f'{title}\npixels={int(pix)} ({ratio*100:.2f}%)', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 最右：统计板
    ax_stat = plt.subplot(2, 8, 7)
    ax_stat.axis('off')
    ax_stat2 = plt.subplot(2, 8, 7 + 8)
    ax_stat2.axis('off')

    cos_sn = sim_metrics.get('cos_student_npz', None)
    cos_st = sim_metrics.get('cos_student_teacher', None)
    cos_nt = sim_metrics.get('cos_npz_teacher', None)
    mae_sn = sim_metrics.get('mae_student_npz', None)
    mae_st = sim_metrics.get('mae_student_teacher', None)
    mae_nt = sim_metrics.get('mae_npz_teacher', None)
    l2_sn = sim_metrics.get('l2_student_npz', None)
    l2_st = sim_metrics.get('l2_student_teacher', None)
    l2_nt = sim_metrics.get('l2_npz_teacher', None)

    iou_sn = mask_ious.get('iou_student_npz', None)
    iou_st = mask_ious.get('iou_student_teacher', None)
    iou_nt = mask_ious.get('iou_npz_teacher', None)

    def _fmt(v):
        return 'NA' if v is None else f'{v:.4f}'

    text_top = (
        'Feature Similarity\n'
        f'Student vs NPZ   Cos:{_fmt(cos_sn)}  MAE:{_fmt(mae_sn)}  L2:{_fmt(l2_sn)}\n'
        f'Student vs Real  Cos:{_fmt(cos_st)}  MAE:{_fmt(mae_st)}  L2:{_fmt(l2_st)}\n'
        f'NPZ vs Real      Cos:{_fmt(cos_nt)}  MAE:{_fmt(mae_nt)}  L2:{_fmt(l2_nt)}\n'
    )
    text_bot = (
        'Mask IOU\n'
        f'Student vs NPZ   {_fmt(iou_sn)}\n'
        f'Student vs Real  {_fmt(iou_st)}\n'
        f'NPZ vs Real      {_fmt(iou_nt)}\n'
    )

    ax_stat.text(0.05, 0.95, text_top, fontsize=9, family='monospace', va='top')
    ax_stat2.text(0.05, 0.95, text_bot, fontsize=9, family='monospace', va='top')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


__all__ = ['draw_panel']



#!/usr/bin/env python3
"""
简单对比：RepViT+Adapter vs SAM2 Native
完全使用SAM2官方流程
"""
import sys
from pathlib import Path
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
import random
import json
from pycocotools import mask as mask_utils
import math

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear
        sys.modules[_mod] = stub_mod

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='outputs/adapter_align_yolov8s_mse_unfreeze_50ep.pth')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 设备: {device}\n")

# ========== 1. 加载SAM2模型 ==========
print("="*60)
print("1. 加载SAM2模型")
print("="*60)

sam2_config_dir = _ROOT / "vfmkd" / "sam2" / "sam2" / "configs"
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
    sam2_model = build_sam2(
        config_file='sam2.1/sam2.1_hiera_b+.yaml',
        ckpt_path='weights/sam2.1_hiera_base_plus.pt',
        device=str(device)
    )

sam2_model.eval()
sam2_model.sam_mask_decoder.use_high_res_features = False
print("[OK] SAM2模型加载完成\n")

# ========== 2. 加载YOLOv8+Adapter ==========
print("="*60)
print("2. 加载YOLOv8+Adapter")
print("="*60)

config = {'model_size': 's', 'pretrained': False, 'freeze_backbone': False}
backbone = YOLOv8Backbone(config).to(device).eval()
in_ch_s16 = backbone.get_feature_dims()[1]  # 获取P4的通道数
adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device).eval()

ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
backbone.load_state_dict(ckpt['backbone'])
adapter.load_state_dict(ckpt['adapter'])
print(f"[OK] YOLOv8+Adapter加载完成: {args.weights}\n")

# 仅使用框提示（中等实例选择）
TARGET_RATIO = 0.10   # 目标实例面积占比（相对于整图）
MIN_RATIO = 0.03      # 最小面积占比阈值（过滤巨大/极小实例）
MAX_RATIO = 0.30      # 最大面积占比阈值
PAD_RATIO = 0.05      # 画框时在实例外接框基础上额外扩展比例

def score_instance_by_bbox(rle, img_w: int, img_h: int, img_area: float) -> tuple:
    """使用RLE的面积与bbox计算打分，返回(score, bbox[x,y,w,h], ratio, fill, aspect).
    - 越接近TARGET_RATIO越好
    - 惩罚极端长宽比与条带（横向/纵向跨满）
    - 偏好较高填充度 fill = area/(w*h)
    不解码整mask，速度更快。
    """
    try:
        area = float(mask_utils.area(rle))
        if area <= 0:
            return (float('inf'), None, 0.0, 0.0, 0.0)
        bbox = mask_utils.toBbox(rle)  # [x, y, w, h]
        x, y, w, h = [float(v) for v in bbox]
        if w <= 1 or h <= 1:
            return (float('inf'), None, 0.0, 0.0, 0.0)
        ratio = area / img_area
        aspect = max(w / h, h / w)
        width_ratio = w / float(img_w)
        height_ratio = h / float(img_h)
        fill = area / (w * h)

        size_penalty = abs(ratio - TARGET_RATIO)
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            size_penalty += 0.5
        compact_penalty = max(0.0, (aspect - 3.0) / 3.0)  # >3 开始惩罚
        stripe_penalty = 0.0
        if width_ratio > 0.8 and height_ratio < 0.25:
            stripe_penalty += 1.0
        if height_ratio > 0.8 and width_ratio < 0.25:
            stripe_penalty += 1.0
        sparsity_penalty = max(0.0, 0.2 - fill)

        score = size_penalty + compact_penalty + stripe_penalty + sparsity_penalty
        return (score, [x, y, w, h], ratio, fill, aspect)
    except Exception:
        return (float('inf'), None, 0.0, 0.0, 0.0)

# ========== 3. 随机选择图像 ==========
print("="*60)
print("3. 随机选择测试图像")
print("="*60)

# 从300张图片中筛选“中等实例（面积占比在[MIN_RATIO, MAX_RATIO]附近）”的5张
data_dir = Path(r'C:\AiBuild\paper\detect\EdgeSAM-master\datasets\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0')
all_images = sorted(list(data_dir.glob('*.jpg')))[:300]
img_scores = []  # (score, in_range, img_path)
for img_path in all_images:
    json_path = img_path.with_suffix('.json')
    try:
        with open(json_path, 'r') as f:
            js = json.load(f)
        h = js['image']['height']
        w = js['image']['width']
        img_area = float(h * w) if h and w else 0.0
        if img_area <= 0:
            continue
        ratios = []
        for ann in js.get('annotations', []):
            try:
                a = float(mask_utils.area(ann['segmentation']))
                r = a / img_area
                ratios.append(r)
            except Exception:
                continue
        if not ratios:
            continue
        # 离TARGET最近的实例占比
        best_r = min(ratios, key=lambda r: abs(r - TARGET_RATIO))
        in_range = (MIN_RATIO <= best_r <= MAX_RATIO)
        score = abs(best_r - TARGET_RATIO)
        # in_range优先，score越小越好
        img_scores.append((score, int(in_range == False), best_r, img_path))
    except Exception:
        continue

# 选取5张（优先范围内，再按接近程度）
img_scores.sort(key=lambda x: (x[1], x[0]))
test_images = [t[3] for t in img_scores[:5]]

print(f"[INFO] 从{len(all_images)}张图片中选取中等实例的5张进行测试：")
for i, img_path in enumerate(test_images, 1):
    print(f"  {i}. {img_path.name}")
print()

# ========== 4. 对每张图片进行测试 ==========
for test_idx, image_path in enumerate(test_images, 1):
    print("="*60)
    print(f"测试图片 {test_idx}/{len(test_images)}: {image_path.name}")
    print("="*60)
    
    # 加载图像
    image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
    image_np = np.array(image_pil)
    
    # 从JSON的RLE mask中随机采样一个真实前景点
    json_path = image_path.with_suffix('.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 从JSON中选取接近TARGET_RATIO且形状合理的实例，提取其外接框（仅框提示）
    if data['annotations']:
        h = int(data['image']['height']); w = int(data['image']['width'])
        img_area = float(h * w) if h and w else 0.0
        best = None
        for ann in data['annotations']:
            s, bbox, r, fill, aspect = score_instance_by_bbox(ann['segmentation'], w, h, img_area)
            if bbox is None:
                continue
            if best is None or s < best[0]:
                best = (s, bbox, r)
        if best is None:
            point_coords = np.array([[512, 512]])
            point_labels = np.array([1])
            box_tensor = None
            print("[WARN] 未找到有效实例，使用中心点")
        else:
            (score_val, bbox, best_r) = best
            x, y, bw, bh = bbox
            sx = 1024.0 / float(w)
            sy = 1024.0 / float(h)
            x0 = int(round(x * sx)); y0 = int(round(y * sy))
            x1 = int(round((x + bw) * sx)); y1 = int(round((y + bh) * sy))
            # 在实例外扩一定边距
            bw = max(1, x1 - x0); bh = max(1, y1 - y0)
            pad_x = int(round(bw * PAD_RATIO)); pad_y = int(round(bh * PAD_RATIO))
            x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
            x1 = min(1023, x1 + pad_x); y1 = min(1023, y1 + pad_y)
            box_scaled = [max(0, min(1023, x0)), max(0, min(1023, y0)),
                          max(0, min(1023, x1)), max(0, min(1023, y1))]
            box_tensor = torch.tensor(box_scaled, dtype=torch.float32, device=device).unsqueeze(0)
            # 关闭点提示
            point_coords = np.zeros((0, 2), dtype=np.int32)
            point_labels = np.zeros((0,), dtype=np.int64)
            print(f"[INFO] 使用中等实例框(缩放): {box_scaled}, 占比{best_r*100:.2f}%, 原图{w}x{h}")
        # 若最大实例为空，已在上方打印并跳过框
    else:
        # 没有标注就用中心点
        point_coords = np.array([[512, 512]])
        point_labels = np.array([1])
        box_tensor = None
        print(f"[WARN] 无标注，使用图像中心点: (512, 512)")
    
    if box_tensor is not None:
        x0, y0, x1, y1 = [int(v) for v in box_tensor[0].tolist()]
        print(f"[INFO] 使用框: [{x0},{y0},{x1},{y1}]\n")
    else:
        print(f"[INFO] 无框提示\n")
    
    # (1, K, 2) & (1, K)；当K=0时传空张量
    if point_coords.size > 0:
        point_coords_torch = torch.from_numpy(point_coords).float().unsqueeze(0).to(device)
        point_labels_torch = torch.from_numpy(point_labels).long().unsqueeze(0).to(device)
        points_tuple = (point_coords_torch, point_labels_torch)
    else:
        points_tuple = None
    
    # ========== SAM2 Native ==========
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        backbone_out_sam2 = sam2_model.image_encoder(image_tensor)
        # 从FPN取64x64特征 (索引2是P4)
        sam2_embedding = backbone_out_sam2['backbone_fpn'][2]
        print(f"[DEBUG] SAM2 embedding shape: {sam2_embedding.shape}")
        
        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=points_tuple,
            boxes=box_tensor,
            masks=None,
        )
        
        masks_sam2, iou_sam2, _, _ = sam2_model.sam_mask_decoder(
            image_embeddings=sam2_embedding,
            image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=False,
            high_res_features=None,
        )
        
        best_idx_sam2 = iou_sam2[0].argmax().item()
        print(f"[SAM2] 最佳IOU: {iou_sam2[0, best_idx_sam2].item():.4f}")
    
    # ========== YOLOv8+Adapter ==========
    with torch.no_grad():
        x = TF.to_tensor(image_pil).unsqueeze(0).to(device)
        feat = backbone(x)
        yolov8_embedding = adapter(feat)
        
        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=points_tuple,
            boxes=box_tensor,
            masks=None,
        )
        
        masks_yolov8, iou_yolov8, _, _ = sam2_model.sam_mask_decoder(
            image_embeddings=yolov8_embedding,
            image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=False,
            high_res_features=None,
        )
        
        best_idx_yolov8 = iou_yolov8[0].argmax().item()
        print(f"[YOLOv8] 最佳IOU: {iou_yolov8[0, best_idx_yolov8].item():.4f}")
    
    # ========== NPZ特征（离线Teacher） ==========
    npz_path = Path('outputs/features_v1_300') / f"{image_path.stem}_features.npz"
    masks_npz = None
    if npz_path.exists():
        npz = np.load(npz_path)
        npz_feat = None
        if 'P4_S16' in npz.files:
            npz_feat = npz['P4_S16']
        elif 'IMAGE_EMB_S16' in npz.files:
            npz_feat = npz['IMAGE_EMB_S16']
        if npz_feat is not None:
            if npz_feat.ndim == 3:
                npz_feat = npz_feat[None, ...]
            npz_t = torch.from_numpy(np.asarray(npz_feat)).float().to(device)
            with torch.no_grad():
                masks_npz, iou_npz, _, _ = sam2_model.sam_mask_decoder(
                    image_embeddings=npz_t,
                    image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=None,
                )
            best_idx_npz = iou_npz[0].argmax().item()
            print(f"[NPZ] 最佳IOU: {iou_npz[0, best_idx_npz].item():.4f}")
        else:
            print(f"[WARN] NPZ缺少P4_S16键: {npz_path}")
            best_idx_npz = 0
    else:
        print(f"[WARN] 缺少NPZ文件: {npz_path}")
        best_idx_npz = 0
    
    # ========== 后处理 ==========
    mask_sam2_1024 = F.interpolate(
        masks_sam2[:, best_idx_sam2:best_idx_sam2+1],
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    )[0, 0].cpu().numpy()
    mask_sam2_binary = (mask_sam2_1024 > 0).astype(np.uint8)
    
    mask_yolov8_1024 = F.interpolate(
        masks_yolov8[:, best_idx_yolov8:best_idx_yolov8+1],
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    )[0, 0].cpu().numpy()
    mask_yolov8_binary = (mask_yolov8_1024 > 0).astype(np.uint8)
    
    white_sam2 = np.sum(mask_sam2_binary)
    white_yolov8 = np.sum(mask_yolov8_binary)
    
    diff = np.abs(mask_yolov8_binary.astype(float) - mask_sam2_binary.astype(float))
    diff_pixels = np.sum(diff > 0)
    consistency = 100 - (diff_pixels / diff.size * 100)
    
    print(f"[SAM2] 白色像素: {white_sam2:,} ({white_sam2/mask_sam2_binary.size*100:.2f}%)")
    print(f"[YOLOv8] 白色像素: {white_yolov8:,} ({white_yolov8/mask_yolov8_binary.size*100:.2f}%)")
    print(f"[对比] 掩码一致性: {consistency:.2f}%\n")
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), constrained_layout=True)
    
    # 原图+提示点
    ax1 = axes[0]
    ax1.imshow(image_np)
    if point_coords.size > 0:
        ax1.scatter(point_coords[:, 0], point_coords[:, 1], c='red', s=80, marker='*', edgecolors='white', linewidths=1.0)
    # 可选绘制框
    if 'box_tensor' in locals() and box_tensor is not None:
        x0, y0, x1, y1 = [int(v) for v in box_tensor[0].tolist()]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='yellow', linewidth=1.5, linestyle='--')
        ax1.add_patch(rect)
    ax1.set_title(f'Original Image\n{image_path.stem}', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # SAM2二值掩码
    ax2 = axes[1]
    ax2.imshow(mask_sam2_binary, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'SAM2 Native\nIOU: {iou_sam2[0, best_idx_sam2].item():.4f}', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # YOLOv8二值掩码
    ax3 = axes[2]
    ax3.imshow(mask_yolov8_binary, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'YOLOv8+Adapter\nIOU: {iou_yolov8[0, best_idx_yolov8].item():.4f}', fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # NPZ二值掩码
    ax_npz = axes[3]
    if masks_npz is not None:
        mask_npz_1024 = F.interpolate(
            masks_npz[:, best_idx_npz:best_idx_npz+1], size=(1024,1024), mode='bilinear', align_corners=False
        )[0,0].cpu().numpy()
        mask_npz_binary = (mask_npz_1024 > 0).astype(np.uint8)
        ax_npz.imshow(mask_npz_binary, cmap='gray', vmin=0, vmax=1)
        ax_npz.set_title(f'NPZ Feature\nIOU: {iou_npz[0, best_idx_npz].item():.4f}', fontsize=10, fontweight='bold')
    else:
        ax_npz.text(0.5, 0.5, 'NPZ Missing', ha='center', va='center')
    ax_npz.axis('off')
    
    # 叠加对比
    overlay_both = image_np.copy()
    overlay_both[mask_sam2_binary > 0] = overlay_both[mask_sam2_binary > 0] * 0.5 + np.array([0, 128, 0])
    overlay_both[mask_yolov8_binary > 0] = overlay_both[mask_yolov8_binary > 0] * 0.5 + np.array([128, 0, 0])
    overlap_mask = (mask_sam2_binary > 0) & (mask_yolov8_binary > 0)
    overlay_both[overlap_mask] = overlay_both[overlap_mask] * 0.5 + np.array([128, 128, 0])
    
    ax4 = axes[4]
    ax4.imshow(np.clip(overlay_both, 0, 255).astype(np.uint8))
    ax4.set_title(f'Overlay Comparison\nConsistency: {consistency:.1f}%', fontsize=10, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle(f'Test Image {test_idx}/{len(test_images)} - {image_path.stem}', fontsize=12, fontweight='bold')
    plt.savefig(f'outputs/comparison_test_{test_idx}_{image_path.stem}.png', dpi=150, bbox_inches='tight')
    print(f"[OK] 可视化保存到: outputs/comparison_test_{test_idx}_{image_path.stem}.png\n")
    plt.close()

print("="*60)
print("所有测试完成！")
print("="*60)


#!/usr/bin/env python3
"""
检查训练NPZ特征与实际SAM2分割特征的相似度
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

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

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

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

# 初始化SAM2官方transform
sam2_transforms = SAM2Transforms(
    resolution=sam2_model.image_size,
    mask_threshold=0.0,
    max_hole_area=0.0,
    max_sprinkle_area=0.0
)
print("[OK] SAM2模型加载完成\n")

# ========== 2. 随机选择测试图片 ==========
print("="*60)
print("2. 随机选择测试图片")
print("="*60)

npz_dir = Path('outputs/features_v1_300')
all_npz = sorted(list(npz_dir.glob('*.npz')))
test_npz = random.sample(all_npz, min(5, len(all_npz)))

print(f"[INFO] 从{len(all_npz)}个NPZ文件中随机选择{len(test_npz)}个进行对比")
for i, npz_path in enumerate(test_npz, 1):
    print(f"  {i}. {npz_path.name}")
print()

# ========== 3. 对比特征 ==========
print("="*60)
print("3. 对比NPZ特征 vs 实际SAM2特征")
print("="*60)

image_dir = Path(r'C:\AiBuild\paper\detect\EdgeSAM-master\datasets\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0')

similarities = []

for idx, npz_path in enumerate(test_npz, 1):
    print(f"\n{'='*60}")
    print(f"测试 {idx}/{len(test_npz)}: {npz_path.stem}")
    print(f"{'='*60}")
    
    # 加载NPZ特征
    npz_data = np.load(npz_path)
    npz_feature = npz_data['P4_S16']  # (256, 64, 64)
    
    if npz_feature.ndim == 3:
        npz_feature = npz_feature[None, ...]  # (1, 256, 64, 64)
    
    npz_feature_torch = torch.from_numpy(npz_feature).float().to(device)
    
    print(f"[NPZ] P4_S16 shape: {npz_feature_torch.shape}")
    print(f"[NPZ] mean={npz_feature_torch.mean().item():.4f}, std={npz_feature_torch.std().item():.4f}")
    print(f"[NPZ] min={npz_feature_torch.min().item():.4f}, max={npz_feature_torch.max().item():.4f}")
    
    # 加载对应图片 (去掉_features后缀)
    image_id = npz_path.stem.replace('_features', '')
    image_path = image_dir / f"{image_id}.jpg"
    
    if not image_path.exists():
        print(f"[WARN] 图片不存在: {image_path}")
        continue
    
    # 用SAM2实时生成特征
    image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
    image_np = np.array(image_pil)
    
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        backbone_out = sam2_model.image_encoder(image_tensor)
        sam2_feature = backbone_out['backbone_fpn'][2]  # P4: (1, 256, 64, 64)
    
    print(f"[SAM2] P4 shape: {sam2_feature.shape}")
    print(f"[SAM2] mean={sam2_feature.mean().item():.4f}, std={sam2_feature.std().item():.4f}")
    print(f"[SAM2] min={sam2_feature.min().item():.4f}, max={sam2_feature.max().item():.4f}")
    
    # 计算相似度
    # 1. 余弦相似度
    npz_flat = npz_feature_torch.flatten()
    sam2_flat = sam2_feature.flatten()
    cos_sim = F.cosine_similarity(npz_flat.unsqueeze(0), sam2_flat.unsqueeze(0)).item()
    
    # 2. L2距离
    l2_dist = torch.norm(npz_feature_torch - sam2_feature, p=2).item()
    
    # 3. 均方误差
    mse = F.mse_loss(npz_feature_torch, sam2_feature).item()
    
    # 4. 平均绝对误差
    mae = torch.mean(torch.abs(npz_feature_torch - sam2_feature)).item()
    
    print(f"\n[对比结果]")
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  L2距离: {l2_dist:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    similarities.append({
        'image_id': image_id,
        'cos_sim': cos_sim,
        'l2_dist': l2_dist,
        'mse': mse,
        'mae': mae
    })

# ========== 4. 统计总结 ==========
print(f"\n{'='*60}")
print("4. 总体统计")
print(f"{'='*60}")

if similarities:
    avg_cos_sim = np.mean([s['cos_sim'] for s in similarities])
    avg_l2 = np.mean([s['l2_dist'] for s in similarities])
    avg_mse = np.mean([s['mse'] for s in similarities])
    avg_mae = np.mean([s['mae'] for s in similarities])
    
    print(f"平均余弦相似度: {avg_cos_sim:.6f}")
    print(f"平均L2距离: {avg_l2:.4f}")
    print(f"平均MSE: {avg_mse:.6f}")
    print(f"平均MAE: {avg_mae:.6f}")
    
    print(f"\n{'='*60}")
    print("结论:")
    print(f"{'='*60}")
    if avg_cos_sim > 0.99:
        print("✅ 特征高度一致 (余弦相似度 > 0.99)")
    elif avg_cos_sim > 0.95:
        print("⚠️ 特征基本一致 (余弦相似度 > 0.95)")
    elif avg_cos_sim > 0.90:
        print("⚠️ 特征存在一定差异 (余弦相似度 > 0.90)")
    else:
        print("❌ 特征差异较大 (余弦相似度 < 0.90)")
    
    print(f"\n说明:")
    print(f"- 训练时使用的NPZ特征与实际分割时SAM2生成的特征相似度为 {avg_cos_sim:.4f}")
    print(f"- 如果相似度较低，说明训练特征提取可能存在问题")
    print(f"- 如果相似度很高，说明YOLOv8+Adapter性能问题不在特征提取环节")

print(f"\n{'='*60}")
print("完成！")
print(f"{'='*60}")


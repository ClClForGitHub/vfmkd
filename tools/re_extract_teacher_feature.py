#!/usr/bin/env python3
"""
重新提取teacher特征 - 直接使用vision_features而不是get_image_embedding()
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("重新提取teacher特征 - 使用正确的vision_features")
print("="*80)

# 加载SAM2模型
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
print(f"[OK] SAM2模型加载完成\n")

# 加载图像
image_path = 'datasets/coco128/images/train2017/000000000009.jpg'
image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
image_np = np.array(image_pil)

print(f"[INFO] 图像: {image_path}")
print(f"[INFO] 尺寸: {image_np.shape}\n")

# 提取特征 - 正确方式
with torch.no_grad():
    # 预处理
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 使用image_encoder直接提取
    backbone_out = sam2_model.image_encoder(image_tensor)
    vision_features = backbone_out['vision_features']  # 这是正确的！
    
    print(f"[INFO] 特征提取完成:")
    print(f"  Shape: {vision_features.shape}")
    print(f"  Mean: {vision_features.mean():.4f}")
    print(f"  Std: {vision_features.std():.4f}")
    print(f"  Range: [{vision_features.min():.4f}, {vision_features.max():.4f}]")

# 保存到NPZ
output_path = 'datasets/coco128/SAM_Cache/000000000009_sam2_features_correct.npz'
np.savez(
    output_path,
    IMAGE_EMB_S16=vision_features.cpu().numpy(),
    image_shape=image_np.shape,
    extraction_method='vision_features_direct'
)

print(f"\n[OK] 特征已保存到: {output_path}")

# 对比旧的NPZ文件
old_npz_path = 'datasets/coco128/SAM_Cache/000000000009_sam2_features.npz'
if Path(old_npz_path).exists():
    print(f"\n对比旧的NPZ文件:")
    print("-"*80)
    old_npz = np.load(old_npz_path)
    if 'IMAGE_EMB_S16' in old_npz:
        old_feat = torch.from_numpy(old_npz['IMAGE_EMB_S16']).to(device)
        print(f"旧特征:")
        print(f"  Shape: {old_feat.shape}")
        print(f"  Mean: {old_feat.mean():.4f}")
        print(f"  Std: {old_feat.std():.4f}")
        print(f"  Range: [{old_feat.min():.4f}, {old_feat.max():.4f}]")
        
        diff = torch.abs(vision_features - old_feat)
        print(f"\n差异:")
        print(f"  Mean: {diff.mean():.4f}")
        print(f"  Max: {diff.max():.4f}")
        
        if diff.max() > 0.1:
            print(f"  ⚠️  特征差异很大！旧NPZ使用了错误的提取方法")
        else:
            print(f"  ✅ 特征基本一致")

print("\n" + "="*80)
print("完成！现在可以用这个正确的NPZ文件重新训练了")
print("="*80)


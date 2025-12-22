#!/usr/bin/env python3
"""
调试FPN特征分辨率
仔细检查SAM2返回的每一层特征的实际尺寸
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
print("调试SAM2 FPN特征分辨率")
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

# 加载图像并resize到1024x1024
image_path = 'datasets/coco128/images/train2017/000000000009.jpg'
image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
image_np = np.array(image_pil)

print(f"输入图像尺寸: {image_np.shape}")

# 提取特征
with torch.no_grad():
    # 预处理
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    print(f"输入tensor shape: {image_tensor.shape}")
    
    # 使用image_encoder
    backbone_out = sam2_model.image_encoder(image_tensor)
    
    print("\n" + "="*80)
    print("backbone_out 的所有键:")
    print("="*80)
    for key in backbone_out.keys():
        print(f"  - {key}")
    
    print("\n" + "="*80)
    print("vision_features:")
    print("="*80)
    vision_features = backbone_out['vision_features']
    print(f"  Shape: {vision_features.shape}")
    print(f"  Type: {type(vision_features)}")
    
    print("\n" + "="*80)
    print("backbone_fpn (FPN金字塔特征):")
    print("="*80)
    fpn_features = backbone_out['backbone_fpn']
    print(f"  Type: {type(fpn_features)}")
    print(f"  Length: {len(fpn_features)}")
    
    for idx, fpn_feat in enumerate(fpn_features):
        if isinstance(fpn_feat, torch.Tensor):
            h, w = fpn_feat.shape[-2:]
            c = fpn_feat.shape[1] if len(fpn_feat.shape) > 1 else 'N/A'
            
            # 计算下采样倍率
            scale = 1024 // h if h > 0 else 'N/A'
            
            print(f"\n  FPN层 {idx}:")
            print(f"    Shape: {fpn_feat.shape}")
            print(f"    空间尺寸: {h}x{w}")
            print(f"    通道数: {c}")
            print(f"    下采样倍率: S{scale} (1024/{h})")
            print(f"    Mean: {fpn_feat.mean():.4f}, Std: {fpn_feat.std():.4f}")
            
            # 判断对应的level
            if h == 256:
                level = "P2"
            elif h == 128:
                level = "P3"
            elif h == 64:
                level = "P4"
            elif h == 32:
                level = "P5"
            else:
                level = f"未知(可能是P{idx})"
            print(f"    对应level: {level}")
        else:
            print(f"\n  FPN层 {idx}: 非Tensor类型 - {type(fpn_feat)}")
    
    print("\n" + "="*80)
    print("vision_pos_enc (位置编码):")
    print("="*80)
    pos_enc = backbone_out['vision_pos_enc']
    print(f"  Type: {type(pos_enc)}")
    print(f"  Length: {len(pos_enc)}")
    for idx, pos in enumerate(pos_enc):
        if isinstance(pos, torch.Tensor):
            print(f"    Pos {idx}: shape={pos.shape}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)


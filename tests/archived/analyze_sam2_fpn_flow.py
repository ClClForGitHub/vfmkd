#!/usr/bin/env python3
"""
深度分析SAM2 FPN流程
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
print("深度分析SAM2 FPN流程")
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

# 获取配置信息
image_encoder = sam2_model.image_encoder
trunk = image_encoder.trunk  # Hiera
neck = image_encoder.neck    # FpnNeck
scalp = image_encoder.scalp

print(f"\n[配置信息]")
print(f"  scalp: {scalp}")
print(f"  Hiera stages: {trunk.stages if hasattr(trunk, 'stages') else 'N/A'}")
print(f"  Hiera stage_ends: {trunk.stage_ends}")
print(f"  Hiera channel_list: {trunk.channel_list}")
print(f"  FPN backbone_channel_list: {neck.backbone_channel_list}")
print(f"  FPN fpn_top_down_levels: {neck.fpn_top_down_levels}")
print(f"  FPN d_model: {neck.d_model}")

# 加载图像
image_path = 'datasets/coco128/images/train2017/000000000009.jpg'
image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
image_np = np.array(image_pil)

print(f"\n[输入图像]")
print(f"  尺寸: {image_np.shape}")

# 提取特征
with torch.no_grad():
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print(f"\n" + "="*80)
    print("步骤1: Hiera Backbone 输出")
    print("="*80)
    
    # 直接调用trunk (Hiera)
    trunk_outputs = trunk(image_tensor)
    print(f"  Hiera输出层数: {len(trunk_outputs)}")
    for idx, feat in enumerate(trunk_outputs):
        h, w = feat.shape[-2:]
        c = feat.shape[1]
        scale = 1024 // h if h > 0 else 'N/A'
        print(f"  Layer {idx}: shape={feat.shape}, 空间={h}x{w}, 通道={c}, 下采样=S{scale}")
    
    print(f"\n" + "="*80)
    print("步骤2: FPN Neck 处理")
    print("="*80)
    
    # 调用neck (FpnNeck)
    fpn_features, fpn_pos = neck(trunk_outputs)
    print(f"  FPN输出层数: {len(fpn_features)}")
    for idx, feat in enumerate(fpn_features):
        h, w = feat.shape[-2:]
        c = feat.shape[1]
        scale = 1024 // h if h > 0 else 'N/A'
        print(f"  FPN Layer {idx}: shape={feat.shape}, 空间={h}x{w}, 通道={c}, 下采样=S{scale}")
    
    print(f"\n" + "="*80)
    print("步骤3: Scalp 裁剪")
    print("="*80)
    
    print(f"  scalp值: {scalp}")
    if scalp > 0:
        print(f"  裁剪前: {len(fpn_features)}层")
        fpn_features_scalped = fpn_features[:-scalp]
        print(f"  裁剪后: {len(fpn_features_scalped)}层")
        print(f"  被丢弃的层:")
        for idx in range(len(fpn_features) - scalp, len(fpn_features)):
            feat = fpn_features[idx]
            h, w = feat.shape[-2:]
            c = feat.shape[1]
            scale = 1024 // h if h > 0 else 'N/A'
            print(f"    Layer {idx}: shape={feat.shape}, 空间={h}x{w}, 下采样=S{scale}")
    else:
        fpn_features_scalped = fpn_features
        print(f"  无裁剪")
    
    print(f"\n" + "="*80)
    print("步骤4: 最终 backbone_fpn 输出")
    print("="*80)
    
    # 使用完整的image_encoder
    backbone_out = image_encoder(image_tensor)
    final_fpn = backbone_out['backbone_fpn']
    vision_features = backbone_out['vision_features']
    
    print(f"  backbone_fpn层数: {len(final_fpn)}")
    for idx, feat in enumerate(final_fpn):
        h, w = feat.shape[-2:]
        c = feat.shape[1]
        scale = 1024 // h if h > 0 else 'N/A'
        print(f"  Layer {idx}: shape={feat.shape}, 空间={h}x{w}, 下采样=S{scale}")
    
    print(f"\n  vision_features: shape={vision_features.shape}")
    
    print(f"\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"  Hiera输出: {len(trunk_outputs)}层 (分辨率从高到低)")
    print(f"  FPN处理后: {len(fpn_features)}层")
    print(f"  Scalp裁剪: 丢弃最低分辨率的{scalp}层")
    print(f"  最终输出: {len(final_fpn)}层")
    print(f"\n  解释:")
    print(f"    - Hiera产生4层特征: 256x256(S4), 128x128(S8), 64x64(S16), 32x32(S32)")
    print(f"    - FPN进行top-down融合")
    print(f"    - scalp=1 丢弃最低分辨率的32x32层")
    print(f"    - 所以最终只有3层: 256x256(S4), 128x128(S8), 64x64(S16)")
    print(f"    - vision_features = 最低分辨率层 = 64x64(S16)")

print("\n" + "="*80)


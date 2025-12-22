#!/usr/bin/env python3
"""
诊断Large模型的特征提取，检查是否取错了下标
对比Large和Base+的所有backbone_fpn层，找出正确的对应关系
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear
        sys.modules[_mod] = _mod

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2 as _build
from vfmkd.sam2.sam2.utils.transforms import SAM2Transforms

device = torch.device('cuda')
config_dir = _ROOT / 'vfmkd' / 'sam2' / 'sam2' / 'configs'

# 查找图片和NPZ文件
image_path = Path('/home/team/zouzhiyuan/dataset/sa1b/sa_40677.jpg')
npz_path = Path('/home/team/zouzhiyuan/dataset/sa1b/extracted/sa_40677_features.npz')

print('='*80)
print('诊断Large模型特征提取 - 检查下标对应关系')
print('='*80)

# 加载图片
image_pil = Image.open(image_path).convert('RGB')
image_np = np.array(image_pil)
sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
image_tensor = sam2_transforms(image_np).unsqueeze(0).to(device)

# 加载Base+模型
print('\n[1] 加载Base+模型并提取所有backbone_fpn层...')
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    sam2_base = _build(
        config_file='sam2.1/sam2.1_hiera_b+.yaml',
        ckpt_path=str(_ROOT / 'weights' / 'sam2.1_hiera_base_plus.pt'),
        device=str(device)
    )
sam2_base.eval()

with torch.no_grad():
    out_base = sam2_base.forward_image(image_tensor)
    fpn_base = out_base['backbone_fpn']
    print(f'  Base+ scalp参数: {sam2_base.image_encoder.scalp}')
    print(f'  Base+ backbone_fpn长度: {len(fpn_base)}')
    for idx, feat in enumerate(fpn_base):
        h, w = feat.shape[-2:]
        scale = 1024 // h if h > 0 else 'N/A'
        print(f'    Base+[{idx}]: shape={feat.shape}, 空间={h}x{w}, 下采样=S{scale}')
        print(f'            mean={feat.mean().item():.6f}, std={feat.std().item():.6f}')

# 加载Large模型
print('\n[2] 加载Large模型并提取所有backbone_fpn层...')
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    sam2_large = _build(
        config_file='sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=str(_ROOT / 'weights' / 'sam2.1_hiera_large.pt'),
        device=str(device)
    )
sam2_large.eval()

with torch.no_grad():
    out_large = sam2_large.forward_image(image_tensor)
    fpn_large = out_large['backbone_fpn']
    print(f'  Large scalp参数: {sam2_large.image_encoder.scalp}')
    print(f'  Large backbone_fpn长度: {len(fpn_large)}')
    for idx, feat in enumerate(fpn_large):
        h, w = feat.shape[-2:]
        scale = 1024 // h if h > 0 else 'N/A'
        print(f'    Large[{idx}]: shape={feat.shape}, 空间={h}x{w}, 下采样=S{scale}')
        print(f'            mean={feat.mean().item():.6f}, std={feat.std().item():.6f}')

# 加载NPZ特征（Base+的P4_S16，64x64）
print('\n[3] 加载NPZ特征（Base+的P4_S16，期望64x64）...')
npz_data = np.load(npz_path, allow_pickle=True)
npz_p4 = None
for key in ['P4_S16', 'IMAGE_EMB_S16']:
    if key in npz_data.files:
        npz_p4 = torch.tensor(npz_data[key], dtype=torch.float32).to(device)
        if npz_p4.dim() == 3:
            npz_p4 = npz_p4.unsqueeze(0)
        print(f'  NPZ特征键: {key}, shape={npz_p4.shape}')
        break

if npz_p4 is None:
    print('  ⚠️  未找到NPZ特征')
    sys.exit(1)

# 计算所有层之间的相似度
print('\n[4] 计算Large各层与Base+ P4[2]的相似度...')
base_p4 = fpn_base[2]  # Base+的P4 (64x64)

def compute_similarity(feat1, feat2, name1, name2):
    """计算两个特征的相似度（自动处理尺寸和通道差异）"""
    # 如果空间尺寸不同，下采样feat1到feat2的尺寸
    if feat1.shape[-2:] != feat2.shape[-2:]:
        feat1_resized = F.interpolate(feat1, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
    else:
        feat1_resized = feat1
    
    # 如果通道数不同，只比较空间位置的平均值（在通道维度求平均）
    if feat1_resized.shape[1] != feat2.shape[1]:
        # 对通道维度求平均，变成单通道
        feat1_resized = feat1_resized.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        feat2_mean = feat2.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        feat2_mean = feat2
    
    feat1_flat = feat1_resized.flatten(1)
    feat2_flat = feat2_mean.flatten(1)
    cosine_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=1).item()
    l2_dist = torch.norm(feat1_flat - feat2_flat, p=2, dim=1).item()
    mae = torch.abs(feat1_resized - feat2_mean).mean().item()
    return cosine_sim, l2_dist, mae

print('\n  对比Large各层与Base+ P4[2] (64x64):')
for idx, feat_large in enumerate(fpn_large):
    cos_sim, l2, mae = compute_similarity(feat_large, base_p4, f'Large[{idx}]', 'Base+[2]')
    h, w = feat_large.shape[-2:]
    scale = 1024 // h if h > 0 else 'N/A'
    print(f'    Large[{idx}] ({h}x{w}, S{scale}) vs Base+[2] (64x64, S16):')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')

# 对比Large的P5[3] (32x32) 和 Base+的P5[3] (32x32)
print('\n[5] 对比Large的P5[3] (32x32) 和 Base+的P5[3] (32x32):')
if len(fpn_large) > 3 and len(fpn_base) > 3:
    large_p5 = fpn_large[3]
    base_p5 = fpn_base[3]
    cos_sim, l2, mae = compute_similarity(large_p5, base_p5, 'Large[3]', 'Base+[3]')
    print(f'    Large[3] (32x32) vs Base+[3] (32x32):')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')

# 额外测试：Large[3]下采样到64x64与Base+[2]比较
print('\n[5.1] 对比Large的P5[3]下采样到64x64 和 Base+的P4[2] (64x64):')
if len(fpn_large) > 3:
    large_p5_down = F.interpolate(large_p5, size=(64, 64), mode='bilinear', align_corners=False)
    cos_sim, l2, mae = compute_similarity(large_p5_down, base_p4, 'Large[3]→64x64', 'Base+[2]')
    print(f'    Large[3] (32x32) -> 下采样到64x64 vs Base+[2] (64x64):')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')
else:
    print('    ⚠️  无法对比（层数不足）')

# 用户要求的测试：Base+的P5[3]上采样到64x64与Large[2]比较
print('\n[5.2] 对比Base+的P5[3]上采样到64x64 和 Large[2] (64x64):')
if len(fpn_base) > 3:
    base_p5_up = F.interpolate(base_p5, size=(64, 64), mode='bilinear', align_corners=False)
    cos_sim, l2, mae = compute_similarity(base_p5_up, fpn_large[2], 'Base+[3]→64x64', 'Large[2]')
    print(f'    Base+[3] (32x32) -> 上采样到64x64 vs Large[2] (64x64):')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')
    
    # 同时对比Base+[3]与Large[3]的直接比较
    print(f'\n    Base+[3] (32x32) vs Large[3] (32x32) [直接对比]:')
    cos_sim, l2, mae = compute_similarity(base_p5, large_p5, 'Base+[3]', 'Large[3]')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')
else:
    print('    ⚠️  无法对比（层数不足）')

# 对比Large的更大分辨率层（P3[1], P2[0]）和Base+的P4[2]
print('\n[6] 对比Large的更大分辨率层与Base+的P4[2]:')
if len(fpn_large) > 1:
    # Large[1] (P3, 128x128) -> 下采样到64x64后与Base+[2]比较
    large_p3 = fpn_large[1]
    print(f'    Large[1] (128x128) -> 下采样到64x64 vs Base+[2] (64x64):')
    large_p3_down = F.interpolate(large_p3, size=(64, 64), mode='bilinear', align_corners=False)
    cos_sim, l2, mae = compute_similarity(large_p3_down, base_p4, 'Large[1]→64x64', 'Base+[2]')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')

if len(fpn_large) > 0:
    # Large[0] (P2, 256x256) -> 下采样到64x64后与Base+[2]比较
    large_p2 = fpn_large[0]
    print(f'    Large[0] (256x256) -> 下采样到64x64 vs Base+[2] (64x64):')
    large_p2_down = F.interpolate(large_p2, size=(64, 64), mode='bilinear', align_corners=False)
    cos_sim, l2, mae = compute_similarity(large_p2_down, base_p4, 'Large[0]→64x64', 'Base+[2]')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')

# 对比NPZ特征与所有Large层
print('\n[7] 对比NPZ特征（Base+的P4_S16）与Large所有层:')
for idx, feat_large in enumerate(fpn_large):
    cos_sim, l2, mae = compute_similarity(feat_large, npz_p4, f'Large[{idx}]', 'NPZ_P4')
    h, w = feat_large.shape[-2:]
    scale = 1024 // h if h > 0 else 'N/A'
    print(f'    Large[{idx}] ({h}x{w}, S{scale}) vs NPZ_P4 (64x64):')
    print(f'      余弦相似度: {cos_sim:.6f}, L2距离: {l2:.6f}, MAE: {mae:.6f}')

print('\n' + '='*80)
print('诊断完成')
print('='*80)


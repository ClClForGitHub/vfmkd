#!/usr/bin/env python3
"""
对比Base+和Large模型的256通道特征
1. 正确加载两个模型的预训练权重
2. 提取所有256通道的特征层
3. 同分辨率的特征互相进行相似度对比
4. 可视化每一层的mean channel特征
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

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

# 查找图片
image_path = Path('/home/team/zouzhiyuan/dataset/sa1b/sa_40677.jpg')
output_dir = Path(_ROOT / 'outputs')
output_dir.mkdir(parents=True, exist_ok=True)

print('='*80)
print('对比Base+和Large模型的256通道特征')
print('='*80)

# 加载图片
print('\n[1] 加载测试图片...')
image_pil = Image.open(image_path).convert('RGB')
image_np = np.array(image_pil)
sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
image_tensor = sam2_transforms(image_np).unsqueeze(0).to(device)
print(f'  图片尺寸: {image_np.shape}')
print(f'  预处理后tensor: {image_tensor.shape}')

# 加载Base+模型（确保正确加载权重）
print('\n[2] 加载Base+模型（验证权重加载）...')
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    sam2_base = _build(
        config_file='sam2.1/sam2.1_hiera_b+.yaml',
        ckpt_path=str(_ROOT / 'weights' / 'sam2.1_hiera_base_plus.pt'),
        device=str(device)
    )
sam2_base.eval()

# 验证权重是否加载
first_param = next(sam2_base.parameters())
print(f'  第一个参数统计: mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}')
if abs(first_param.mean().item()) < 0.1 and first_param.std().item() > 0.1:
    print('  ✓ 权重统计正常，预训练权重已加载')
else:
    print('  ⚠️  警告: 权重统计异常，可能未正确加载')

# 加载Large模型（确保正确加载权重）
print('\n[3] 加载Large模型（验证权重加载）...')
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    sam2_large = _build(
        config_file='sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path=str(_ROOT / 'weights' / 'sam2.1_hiera_large.pt'),
        device=str(device)
    )
sam2_large.eval()

# 验证权重是否加载
first_param = next(sam2_large.parameters())
print(f'  第一个参数统计: mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}')
if abs(first_param.mean().item()) < 0.1 and first_param.std().item() > 0.1:
    print('  ✓ 权重统计正常，预训练权重已加载')
else:
    print('  ⚠️  警告: 权重统计异常，可能未正确加载')

# 提取特征
print('\n[4] 提取backbone_fpn特征...')
with torch.no_grad():
    out_base = sam2_base.forward_image(image_tensor)
    out_large = sam2_large.forward_image(image_tensor)
    
    fpn_base = out_base['backbone_fpn']
    fpn_large = out_large['backbone_fpn']

print(f'  Base+ backbone_fpn长度: {len(fpn_base)}')
print(f'  Large backbone_fpn长度: {len(fpn_large)}')

# 收集所有层的信息（不只是256通道的）
print('\n[5] 收集所有backbone_fpn层信息...')
print('\n  检查forward_image是否对某些层做了额外处理:')
print(f'  Base+ use_high_res_features_in_sam: {sam2_base.use_high_res_features_in_sam}')
print(f'  Large use_high_res_features_in_sam: {sam2_large.use_high_res_features_in_sam}')

all_features_base = {}
all_features_large = {}

for idx, feat in enumerate(fpn_base):
    h, w = feat.shape[-2:]
    c = feat.shape[1]
    processed = ''
    if sam2_base.use_high_res_features_in_sam and idx in [0, 1]:
        processed = f' [已通过conv_s{idx}处理]'
    all_features_base[f'{idx}_{h}x{w}'] = {
        'feature': feat,
        'idx': idx,
        'shape': feat.shape,
        'resolution': (h, w),
        'channels': c,
        'mean': feat.mean().item(),
        'std': feat.std().item(),
        'processed': processed
    }
    print(f'  Base+[{idx}]: shape={feat.shape}, 分辨率={h}x{w}, 通道={c}{processed}, mean={feat.mean().item():.6f}, std={feat.std().item():.6f}')

for idx, feat in enumerate(fpn_large):
    h, w = feat.shape[-2:]
    c = feat.shape[1]
    processed = ''
    if sam2_large.use_high_res_features_in_sam and idx in [0, 1]:
        processed = f' [已通过conv_s{idx}处理]'
    all_features_large[f'{idx}_{h}x{w}'] = {
        'feature': feat,
        'idx': idx,
        'shape': feat.shape,
        'resolution': (h, w),
        'channels': c,
        'mean': feat.mean().item(),
        'std': feat.std().item(),
        'processed': processed
    }
    print(f'  Large[{idx}]: shape={feat.shape}, 分辨率={h}x{w}, 通道={c}{processed}, mean={feat.mean().item():.6f}, std={feat.std().item():.6f}')

# 筛选256通道的特征（用于后续对比）
features_base_256 = {k: v for k, v in all_features_base.items() if v['channels'] == 256}
features_large_256 = {k: v for k, v in all_features_large.items() if v['channels'] == 256}

# 计算同分辨率特征的相似度
print('\n[6] 计算同分辨率特征的相似度...')

def compute_similarity(feat1, feat2):
    """计算两个特征的相似度"""
    feat1_flat = feat1.flatten(1)
    feat2_flat = feat2.flatten(1)
    cosine_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=1).item()
    l2_dist = torch.norm(feat1_flat - feat2_flat, p=2, dim=1).item()
    mae = torch.abs(feat1 - feat2).mean().item()
    mse = F.mse_loss(feat1, feat2).item()
    rmse = np.sqrt(mse)
    return {
        'cosine_sim': cosine_sim,
        'l2_dist': l2_dist,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

# 对比所有同分辨率的层（不限于256通道）
comparisons = []
comparisons_all_layers = []

print('\n  对比所有同分辨率的层（包括不同通道数）:')
for key_base, feat_info_base in all_features_base.items():
    res_base = feat_info_base['resolution']
    for key_large, feat_info_large in all_features_large.items():
        res_large = feat_info_large['resolution']
        if res_base == res_large:
            # 同分辨率，计算相似度（自动处理通道数差异）
            sim = compute_similarity(feat_info_base['feature'], feat_info_large['feature'])
            comp_info = {
                'base_idx': feat_info_base['idx'],
                'large_idx': feat_info_large['idx'],
                'resolution': res_base,
                'base_channels': feat_info_base['channels'],
                'large_channels': feat_info_large['channels'],
                'similarity': sim,
                'base_processed': feat_info_base['processed'],
                'large_processed': feat_info_large['processed']
            }
            comparisons_all_layers.append(comp_info)
            
            if feat_info_base['channels'] == 256 and feat_info_large['channels'] == 256:
                comparisons.append(comp_info)
            
            print(f'\n  对比: Base+[{feat_info_base["idx"]}] ({res_base[0]}x{res_base[1]}, {feat_info_base["channels"]}ch) vs Large[{feat_info_large["idx"]}] ({res_large[0]}x{res_large[1]}, {feat_info_large["channels"]}ch)')
            print(f'    余弦相似度: {sim["cosine_sim"]:.6f}')
            print(f'    L2距离: {sim["l2_dist"]:.6f}')
            print(f'    MAE: {sim["mae"]:.6f}')
            print(f'    RMSE: {sim["rmse"]:.6f}')

# 可视化：将[0]和[1]层下采样到64x64，与[2]层对比
print('\n[7] 生成可视化（下采样对比）...')

# 下采样到64x64用于对比
target_size = (64, 64)

# 准备所有64x64的特征（原始+下采样）
features_64x64_base = []
features_64x64_large = []

# [2]层：原始64x64
if 2 in [info['idx'] for info in all_features_base.values()]:
    for key, info in all_features_base.items():
        if info['idx'] == 2:
            features_64x64_base.append({
                'feature': info['feature'],
                'label': f'[2] Original 64x64 ({info["channels"]}ch)',
                'source': 'original'
            })
            break

if 2 in [info['idx'] for info in all_features_large.values()]:
    for key, info in all_features_large.items():
        if info['idx'] == 2:
            features_64x64_large.append({
                'feature': info['feature'],
                'label': f'[2] Original 64x64 ({info["channels"]}ch)',
                'source': 'original'
            })
            break

# [0]层：256x256下采样到64x64
if 0 in [info['idx'] for info in all_features_base.values()]:
    for key, info in all_features_base.items():
        if info['idx'] == 0:
            feat_downsampled = F.interpolate(
                info['feature'], 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            features_64x64_base.append({
                'feature': feat_downsampled,
                'label': f'[0] 256x256→64x64 ({info["channels"]}ch)',
                'source': 'downsampled'
            })
            break

if 0 in [info['idx'] for info in all_features_large.values()]:
    for key, info in all_features_large.items():
        if info['idx'] == 0:
            feat_downsampled = F.interpolate(
                info['feature'], 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            features_64x64_large.append({
                'feature': feat_downsampled,
                'label': f'[0] 256x256→64x64 ({info["channels"]}ch)',
                'source': 'downsampled'
            })
            break

# [1]层：128x128下采样到64x64
if 1 in [info['idx'] for info in all_features_base.values()]:
    for key, info in all_features_base.items():
        if info['idx'] == 1:
            feat_downsampled = F.interpolate(
                info['feature'], 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            features_64x64_base.append({
                'feature': feat_downsampled,
                'label': f'[1] 128x128→64x64 ({info["channels"]}ch)',
                'source': 'downsampled'
            })
            break

if 1 in [info['idx'] for info in all_features_large.values()]:
    for key, info in all_features_large.items():
        if info['idx'] == 1:
            feat_downsampled = F.interpolate(
                info['feature'], 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            features_64x64_large.append({
                'feature': feat_downsampled,
                'label': f'[1] 128x128→64x64 ({info["channels"]}ch)',
                'source': 'downsampled'
            })
            break

# 创建可视化
num_features = max(len(features_64x64_base), len(features_64x64_large))
rows = num_features
cols = 2  # Base+和Large各一列

fig = plt.figure(figsize=(16, 4 * rows))
plt.suptitle('Base+ vs Large: 64x64 Feature Comparison\n(Original [2] vs Downsampled [0] and [1])', fontsize=16, fontweight='bold')

plot_idx = 1
for i in range(num_features):
    # Base+特征
    if i < len(features_64x64_base):
        feat_info_base = features_64x64_base[i]
        ax1 = plt.subplot(rows, cols, plot_idx)
        feat_base = feat_info_base['feature'][0]  # (C, H, W)
        feat_base_mean = feat_base.mean(dim=0).cpu().numpy()  # (H, W)
        im1 = ax1.imshow(feat_base_mean, cmap='viridis')
        title = f'Base+ {feat_info_base["label"]}\nmean={feat_base_mean.mean():.4f}, std={feat_base_mean.std():.4f}'
        ax1.set_title(title, fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
    
    # Large特征
    if i < len(features_64x64_large):
        feat_info_large = features_64x64_large[i]
        ax2 = plt.subplot(rows, cols, plot_idx + 1)
        feat_large = feat_info_large['feature'][0]  # (C, H, W)
        feat_large_mean = feat_large.mean(dim=0).cpu().numpy()  # (H, W)
        im2 = ax2.imshow(feat_large_mean, cmap='viridis')
        title = f'Large {feat_info_large["label"]}\nmean={feat_large_mean.mean():.4f}, std={feat_large_mean.std():.4f}'
        ax2.set_title(title, fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # 如果两者都存在，计算相似度
        if i < len(features_64x64_base):
            sim = compute_similarity(feat_info_base['feature'], feat_info_large['feature'])
            ax2.text(0.02, 0.98, 
                    f'Cosine: {sim["cosine_sim"]:.4f}\nMAE: {sim["mae"]:.4f}',
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_idx += cols

plt.tight_layout()
output_path = output_dir / 'base_large_256ch_features_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'  [OK] 可视化保存至: {output_path}')

# 生成详细对比报告
print('\n[8] 生成详细对比报告...')
report_lines = [
    '='*80,
    'Base+ vs Large: 256通道特征详细对比报告',
    '='*80,
    '',
    f'测试图片: {image_path.name}',
    f'图片尺寸: {image_np.shape}',
    '',
    '模型信息:',
    f'  Base+模型: sam2.1_hiera_base_plus.pt',
    f'  Large模型: sam2.1_hiera_large.pt',
    '',
    f'  Base+ use_high_res_features_in_sam: {sam2_base.use_high_res_features_in_sam}',
    f'  Large use_high_res_features_in_sam: {sam2_large.use_high_res_features_in_sam}',
    '',
    '所有特征层信息:',
]

for key, info in sorted(all_features_base.items(), key=lambda x: x[1]['idx']):
    report_lines.append(f'  Base+[{info["idx"]}]: {info["shape"]}, 分辨率={info["resolution"]}, 通道={info["channels"]}, mean={info["mean"]:.6f}, std={info["std"]:.6f}{info["processed"]}')

for key, info in sorted(all_features_large.items(), key=lambda x: x[1]['idx']):
    report_lines.append(f'  Large[{info["idx"]}]: {info["shape"]}, 分辨率={info["resolution"]}, 通道={info["channels"]}, mean={info["mean"]:.6f}, std={info["std"]:.6f}{info["processed"]}')

report_lines.extend([
    '',
    '所有同分辨率特征相似度对比:',
])

for comp in comparisons_all_layers:
    sim = comp['similarity']
    report_lines.extend([
        f'  Base+[{comp["base_idx"]}] ({comp["resolution"][0]}x{comp["resolution"][1]}, {comp["base_channels"]}ch{comp["base_processed"]}) vs Large[{comp["large_idx"]}] ({comp["resolution"][0]}x{comp["resolution"][1]}, {comp["large_channels"]}ch{comp["large_processed"]}):',
        f'    余弦相似度: {sim["cosine_sim"]:.6f}',
        f'    L2距离: {sim["l2_dist"]:.6f}',
        f'    MAE: {sim["mae"]:.6f}',
        f'    MSE: {sim["mse"]:.6f}',
        f'    RMSE: {sim["rmse"]:.6f}',
        ''
    ])

report_lines.append('='*80)

report_text = '\n'.join(report_lines)
report_path = output_dir / 'base_large_256ch_features_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f'  [OK] 报告保存至: {report_path}')
print(report_text)

print('\n' + '='*80)
print('对比完成！')
print('='*80)


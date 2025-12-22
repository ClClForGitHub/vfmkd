#!/usr/bin/env python3
"""
多模型(SAM2 tiny/small/base+/large)特征对比可视化：
- 正确加载各自权重
- 提取backbone_fpn各层
- 统一对比项：
  [0] 256x256 → 下采样到64x64
  [1] 128x128 → 下采样到64x64
  [2] 64x64   → 原始
- 将4个模型按列排列，3个对比项按行排列，展示mean channel，并在每格标注mean/std
"""
import sys
from pathlib import Path
import types as _types
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

# stub optional deps
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
weights_dir = _ROOT / 'weights'
output_dir = _ROOT / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

models = [
    {"name": "tiny", "cfg": 'sam2.1/sam2.1_hiera_t.yaml', "ckpt": 'sam2.1_hiera_tiny.pt'},
    {"name": "small", "cfg": 'sam2.1/sam2.1_hiera_s.yaml', "ckpt": 'sam2.1_hiera_small.pt'},
    {"name": "base_plus", "cfg": 'sam2.1/sam2.1_hiera_b+.yaml', "ckpt": 'sam2.1_hiera_base_plus.pt'},
    {"name": "large", "cfg": 'sam2.1/sam2.1_hiera_l.yaml', "ckpt": 'sam2.1_hiera_large.pt'},
]

# 1) 读图
image_path = Path('/home/team/zouzhiyuan/dataset/sa1b/sa_40677.jpg')
img = Image.open(image_path).convert('RGB')
img_np = np.array(img)
trans = SAM2Transforms(resolution=1024, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
img_tensor = trans(img_np).unsqueeze(0).to(device)

# 2) 加载各模型并提取需要的层
def load_model(cfg_file: str, ckpt_name: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        model = _build(
            config_file=cfg_file,
            ckpt_path=str(weights_dir / ckpt_name),
            device=str(device)
        )
    model.eval()
    return model

all_feats = {}
for m in models:
    sam2 = load_model(m['cfg'], m['ckpt'])
    with torch.no_grad():
        out = sam2.forward_image(img_tensor)
        fpn = out['backbone_fpn']
    # 收集三项：0->64, 1->64, 2原始(若存在)
    feats_64 = []
    labels_64 = []
    # [0]
    if len(fpn) > 0:
        f0 = fpn[0]
        f0d = F.interpolate(f0, size=(64, 64), mode='bilinear', align_corners=False)
        feats_64.append(f0d)
        labels_64.append('[0] 256→64')
    # [1]
    if len(fpn) > 1:
        f1 = fpn[1]
        f1d = F.interpolate(f1, size=(64, 64), mode='bilinear', align_corners=False)
        feats_64.append(f1d)
        labels_64.append('[1] 128→64')
    # [2]
    if len(fpn) > 2:
        f2 = fpn[2]
        feats_64.append(f2)
        labels_64.append('[2] 64')

    all_feats[m['name']] = {
        'feats': feats_64,
        'labels': labels_64,
    }

# 3) 组合可视化：行=对比项(最多3)，列=模型数
max_rows = max(len(v['feats']) for v in all_feats.values())
cols = len(models)
rows = max_rows
fig = plt.figure(figsize=(4*cols, 4*rows))
plt.suptitle('SAM2 多模型 64x64 特征对比 (Mean Channel)', fontsize=14, fontweight='bold')

for col, m in enumerate(models):
    name = m['name']
    feats = all_feats[name]['feats']
    labels = all_feats[name]['labels']
    for row in range(rows):
        ax_idx = row*cols + col + 1
        ax = plt.subplot(rows, cols, ax_idx)
        if row < len(feats):
            feat = feats[row][0]  # (C,H,W)
            feat_mean = feat.mean(dim=0).cpu().numpy()
            im = ax.imshow(feat_mean, cmap='viridis')
            ax.set_title(f'{name}\n{labels[row]}\nmean={feat_mean.mean():.4f}, std={feat_mean.std():.4f}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

plt.tight_layout()
output_path = output_dir / 'multi_models_64x64_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'[OK] 合并可视化已保存: {output_path}')

# 4) 生成原生分辨率 4x4 合图（每模型一列：256x256, 128x128, 64x64, 32x32）
print('[INFO] 生成原生分辨率 4x4 合图...')
fig2 = plt.figure(figsize=(4*len(models), 4*4))
plt.suptitle('SAM2 多模型 原生分辨率特征 (Mean Channel) — 行:256/128/64/32, 列:tiny/small/base+/large', fontsize=12, fontweight='bold')

# 重新提取一次原始 fpn，按分辨率绘制
native_fpn = {}
for m in models:
    name = m['name']
    sam2 = load_model(m['cfg'], m['ckpt'])
    with torch.no_grad():
        out = sam2.forward_image(img_tensor)
        native_fpn[name] = out['backbone_fpn']

res_order = ['256x256', '128x128', '64x64', '32x32']
res_to_idx = {}
# 依据 shape 匹配 idx
for name, fpn in native_fpn.items():
    idx_map = {}
    for i, feat in enumerate(fpn):
        h, w = feat.shape[-2:]
        key = f'{h}x{w}'
        idx_map[key] = i
    res_to_idx[name] = idx_map

for row, res in enumerate(res_order):
    for col, m in enumerate(models):
        name = m['name']
        ax = plt.subplot(4, len(models), row*len(models) + col + 1)
        idx = res_to_idx[name].get(res, None)
        if idx is None:
            ax.axis('off')
            continue
        feat = native_fpn[name][idx][0]
        feat_mean = feat.mean(dim=0).cpu().numpy()
        im = ax.imshow(feat_mean, cmap='viridis')
        ax.set_title(f'{name} [{idx}] {res}\nmean={feat_mean.mean():.4f}, std={feat_mean.std():.4f}', fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
output_path2 = output_dir / 'multi_models_native_4x4.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f'[OK] 原生分辨率合图已保存: {output_path2}')

# 5) 权重差异检查：sam_prompt_encoder 与 sam_mask_decoder
print('[INFO] 检查 sam_prompt_encoder / sam_mask_decoder 权重差异...')
encoders = {}
decoders = {}
for m in models:
    name = m['name']
    sam2 = load_model(m['cfg'], m['ckpt'])
    encoders[name] = sam2.sam_prompt_encoder.state_dict()
    decoders[name] = sam2.sam_mask_decoder.state_dict()

def summarize_state_dict(sd):
    total_params = 0
    stats = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        total_params += v.numel()
        stats[k] = (tuple(v.shape), v.float().mean().item(), v.float().std().item())
    return total_params, stats

# 比较 pairwise 是否一致（形状一致但数值不同也要标出）
report_lines = []
report_lines.append('====== SAM2 Encoders/Decoders 权重对比 ======')

# Encoders
report_lines.append('\n[Prompt Encoder] 对比:')
base_name = 'base_plus'
for name in [m['name'] for m in models if m['name'] != base_name]:
    t_base, s_base = summarize_state_dict(encoders[base_name])
    t_cur, s_cur = summarize_state_dict(encoders[name])
    same_keys = set(s_base.keys()) & set(s_cur.keys())
    diff_count = 0
    shape_diff = 0
    for k in same_keys:
        if s_base[k][0] != s_cur[k][0]:
            shape_diff += 1
        else:
            # 形状一致，检查均值是否几乎相等（粗略判断）
            if abs(s_base[k][1] - s_cur[k][1]) > 1e-6 or abs(s_base[k][2] - s_cur[k][2]) > 1e-6:
                diff_count += 1
    report_lines.append(f'  {base_name} vs {name}: keys={len(same_keys)}, shape_diff={shape_diff}, stat_diff={diff_count}, params({base_name})={t_base}, params({name})={t_cur}')

# Decoders
report_lines.append('\n[Mask Decoder] 对比:')
for name in [m['name'] for m in models if m['name'] != base_name]:
    t_base, s_base = summarize_state_dict(decoders[base_name])
    t_cur, s_cur = summarize_state_dict(decoders[name])
    same_keys = set(s_base.keys()) & set(s_cur.keys())
    diff_count = 0
    shape_diff = 0
    for k in same_keys:
        if s_base[k][0] != s_cur[k][0]:
            shape_diff += 1
        else:
            if abs(s_base[k][1] - s_cur[k][1]) > 1e-6 or abs(s_base[k][2] - s_cur[k][2]) > 1e-6:
                diff_count += 1
    report_lines.append(f'  {base_name} vs {name}: keys={len(same_keys)}, shape_diff={shape_diff}, stat_diff={diff_count}, params({base_name})={t_base}, params({name})={t_cur}')

report_path = output_dir / 'multi_models_weight_diff_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print('[OK] 权重对比报告:', report_path)

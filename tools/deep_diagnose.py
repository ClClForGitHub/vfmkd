#!/usr/bin/env python3
"""
深度诊断：RepViT特征 vs SAM2特征
检查每个环节的差异
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
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

config = {'arch': 'm1', 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None}
backbone = RepViTBackbone(config).to(device).eval()
adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256).to(device).eval()

# 加载MSE训练的权重
ckpt = torch.load('outputs/overfit_mse_weights.pth', map_location='cpu', weights_only=False)
backbone.load_state_dict(ckpt['backbone'])
adapter.load_state_dict(ckpt['adapter'])

# 加载图像
image_path = 'datasets/coco128/images/train2017/000000000009.jpg'
image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
image_np = np.array(image_pil)

# 加载训练时的teacher特征
npz = np.load('datasets/coco128/SAM_Cache/000000000009_sam2_features.npz')
if 'IMAGE_EMB_S16' in npz:
    teacher_feat_saved = torch.from_numpy(npz['IMAGE_EMB_S16']).to(device)
else:
    teacher_feat_saved = torch.from_numpy(npz['image_embedding']).to(device)

print("="*80)
print("深度诊断：特征对比")
print("="*80)

with torch.no_grad():
    # SAM2 Native编码
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    backbone_out_sam2 = sam2_model.image_encoder(image_tensor)
    sam2_embedding = backbone_out_sam2['vision_features']
    
    # RepViT编码
    repvit_input = TF.to_tensor(image_pil).unsqueeze(0).to(device)
    repvit_backbone_feat = backbone(repvit_input)
    repvit_embedding = adapter(repvit_backbone_feat)

print("\n1. 检查teacher特征是否一致:")
print("-"*80)
print(f"SAM2实时编码: shape={sam2_embedding.shape}, mean={sam2_embedding.mean():.4f}, std={sam2_embedding.std():.4f}")
print(f"NPZ保存特征: shape={teacher_feat_saved.shape}, mean={teacher_feat_saved.mean():.4f}, std={teacher_feat_saved.std():.4f}")

diff_teacher = torch.abs(sam2_embedding - teacher_feat_saved)
print(f"差异: mean={diff_teacher.mean():.6f}, max={diff_teacher.max():.6f}")

if diff_teacher.max() < 1e-3:
    print("✅ 训练用的teacher特征与实时SAM2编码一致")
else:
    print("⚠️  训练用的teacher特征与实时SAM2编码有差异！这可能是问题所在！")

print("\n2. 学生特征统计:")
print("-"*80)
print(f"RepViT: mean={repvit_embedding.mean():.4f}, std={repvit_embedding.std():.4f}")
print(f"        range=[{repvit_embedding.min():.4f}, {repvit_embedding.max():.4f}]")
print(f"SAM2:   mean={sam2_embedding.mean():.4f}, std={sam2_embedding.std():.4f}")
print(f"        range=[{sam2_embedding.min():.4f}, {sam2_embedding.max():.4f}]")

feat_diff = torch.abs(repvit_embedding - sam2_embedding)
cosine_sim = F.cosine_similarity(repvit_embedding.flatten(), sam2_embedding.flatten(), dim=0)
print(f"\n差异: mean={feat_diff.mean():.4f}, max={feat_diff.max():.4f}")
print(f"余弦相似度: {cosine_sim:.6f}")

print("\n3. 通道级别分析（前10个通道）:")
print("-"*80)
for ch in range(min(10, 256)):
    repvit_ch = repvit_embedding[0, ch]
    sam2_ch = sam2_embedding[0, ch]
    
    ch_diff = torch.abs(repvit_ch - sam2_ch).mean()
    ch_cosine = F.cosine_similarity(repvit_ch.flatten(), sam2_ch.flatten(), dim=0)
    
    print(f"Ch{ch:3d}: RepViT mean={repvit_ch.mean():7.4f}, SAM2 mean={sam2_ch.mean():7.4f}, "
          f"diff={ch_diff:.4f}, cosine={ch_cosine:.4f}")

print("\n4. 空间分布分析:")
print("-"*80)
# 计算每个空间位置的平均激活
repvit_spatial = repvit_embedding.mean(dim=1)[0]  # (64, 64)
sam2_spatial = sam2_embedding.mean(dim=1)[0]  # (64, 64)

print(f"RepViT空间激活: mean={repvit_spatial.mean():.4f}, std={repvit_spatial.std():.4f}")
print(f"SAM2空间激活:   mean={sam2_spatial.mean():.4f}, std={sam2_spatial.std():.4f}")

# 找到最大激活的位置
repvit_max_pos = torch.argmax(repvit_spatial)
sam2_max_pos = torch.argmax(sam2_spatial)
repvit_max_y, repvit_max_x = repvit_max_pos // 64, repvit_max_pos % 64
sam2_max_y, sam2_max_x = sam2_max_pos // 64, sam2_max_pos % 64

print(f"RepViT最大激活位置: ({repvit_max_y}, {repvit_max_x}), 值={repvit_spatial[repvit_max_y, repvit_max_x]:.4f}")
print(f"SAM2最大激活位置:   ({sam2_max_y}, {sam2_max_x}), 值={sam2_spatial[sam2_max_y, sam2_max_x]:.4f}")

print("\n5. 检查是否存在缩放/偏移问题:")
print("-"*80)
# 尝试找到最佳的缩放因子
scale_factor = (sam2_embedding.std() / repvit_embedding.std()).item()
bias_factor = (sam2_embedding.mean() - repvit_embedding.mean()).item()

print(f"建议的缩放因子 (std比): {scale_factor:.4f}")
print(f"建议的偏移量 (mean差): {bias_factor:.4f}")

# 应用缩放和偏移
repvit_scaled = repvit_embedding * scale_factor + bias_factor
scaled_diff = torch.abs(repvit_scaled - sam2_embedding)
print(f"应用缩放后的差异: mean={scaled_diff.mean():.4f}, max={scaled_diff.max():.4f}")

print("\n6. 可视化特征图差异:")
print("-"*80)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：空间激活图
axes[0, 0].imshow(repvit_spatial.cpu().numpy(), cmap='viridis')
axes[0, 0].set_title('RepViT Spatial Activation')
axes[0, 0].axis('off')

axes[0, 1].imshow(sam2_spatial.cpu().numpy(), cmap='viridis')
axes[0, 1].set_title('SAM2 Spatial Activation')
axes[0, 1].axis('off')

spatial_diff = torch.abs(repvit_spatial - sam2_spatial).cpu().numpy()
im2 = axes[0, 2].imshow(spatial_diff, cmap='hot')
axes[0, 2].set_title(f'Spatial Diff (mean={spatial_diff.mean():.4f})')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# 显示原图（标记prompt点）
axes[0, 3].imshow(image_np)
axes[0, 3].scatter([250], [400], c='red', s=100, marker='*')
axes[0, 3].set_title('Original Image + Prompt')
axes[0, 3].axis('off')

# 第二行：单个通道对比（选择方差最大的通道）
repvit_var = repvit_embedding[0].var(dim=(1, 2))
max_var_ch = repvit_var.argmax().item()

axes[1, 0].imshow(repvit_embedding[0, max_var_ch].cpu().numpy(), cmap='RdBu_r')
axes[1, 0].set_title(f'RepViT Ch{max_var_ch}')
axes[1, 0].axis('off')

axes[1, 1].imshow(sam2_embedding[0, max_var_ch].cpu().numpy(), cmap='RdBu_r')
axes[1, 1].set_title(f'SAM2 Ch{max_var_ch}')
axes[1, 1].axis('off')

ch_diff = torch.abs(repvit_embedding[0, max_var_ch] - sam2_embedding[0, max_var_ch]).cpu().numpy()
im5 = axes[1, 2].imshow(ch_diff, cmap='hot')
axes[1, 2].set_title(f'Ch{max_var_ch} Diff')
axes[1, 2].axis('off')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

# 直方图对比
axes[1, 3].hist(repvit_embedding.cpu().flatten().numpy(), bins=100, alpha=0.5, label='RepViT', color='red')
axes[1, 3].hist(sam2_embedding.cpu().flatten().numpy(), bins=100, alpha=0.5, label='SAM2', color='blue')
axes[1, 3].set_title('Feature Distribution')
axes[1, 3].legend()
axes[1, 3].set_xlabel('Value')
axes[1, 3].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/deep_diagnosis.png', dpi=150, bbox_inches='tight')
print("✅ 可视化保存到: outputs/deep_diagnosis.png")

print("\n" + "="*80)
print("诊断完成！")
print("="*80)


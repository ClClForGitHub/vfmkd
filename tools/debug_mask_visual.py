#!/usr/bin/env python3
"""调试掩码可视化"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载之前保存的掩码（如果有的话）或者重新生成简单测试
# 创建一个测试掩码
mask = torch.zeros(1, 1, 256, 256)
mask[0, 0, 100:150, 100:150] = 1.0  # 创建一个白色方块

# 上采样到1024x1024
mask_upscaled = F.interpolate(mask, size=(1024, 1024), mode='bilinear', align_corners=False)
mask_binary = (mask_upscaled > 0).float()

# 转换为numpy
mask_np = mask_binary[0, 0].cpu().numpy()

print(f"掩码shape: {mask_np.shape}")
print(f"掩码范围: [{mask_np.min()}, {mask_np.max()}]")
print(f"白色像素数: {np.sum(mask_np > 0.5)}")
print(f"白色像素比例: {np.sum(mask_np > 0.5) / mask_np.size * 100:.2f}%")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 灰度图
axes[0].imshow(mask_np, cmap='gray')
axes[0].set_title('Grayscale (gray cmap)')
axes[0].axis('off')

# 二值图
axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Binary (gray cmap, vmin=0, vmax=1)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/debug_mask_test.png', dpi=150, bbox_inches='tight')
print("测试图保存到: outputs/debug_mask_test.png")

# 另一个测试：加载实际生成的掩码查看
print("\n" + "="*80)
print("现在测试实际的掩码输出...")

# 手动运行一次掩码生成看看实际输出
import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
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
import torchvision.transforms.functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载模型
config = {'arch': 'm1', 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None}
backbone = RepViTBackbone(config).to(device).eval()
adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256).to(device).eval()

ckpt = torch.load('outputs/overfit_single_image_weights.pth', map_location='cpu', weights_only=False)
backbone.load_state_dict(ckpt['backbone'])
adapter.load_state_dict(ckpt['adapter'])

# 加载图片
img = Image.open('datasets/coco128/images/train2017/000000000009.jpg').convert('RGB').resize((1024, 1024))
x = TF.to_tensor(img).unsqueeze(0).to(device)

# 前向
with torch.no_grad():
    feat = backbone(x)
    adapted = adapter(feat)
    
print(f"\nAdapter输出shape: {adapted.shape}")
print(f"Adapter统计: mean={adapted.mean().item():.4f}, std={adapted.std().item():.4f}")

# 查看实际掩码的统计
print("\n加载NPZ查看教师特征...")
teacher_data = np.load('datasets/coco128/SAM_Cache/000000000009_sam2_features.npz')
teacher_feat = torch.from_numpy(teacher_data['IMAGE_EMB_S16']).float().to(device)
if teacher_feat.dim() == 5:
    teacher_feat = teacher_feat.squeeze(1)
print(f"教师特征shape: {teacher_feat.shape}")
print(f"教师统计: mean={teacher_feat.mean().item():.4f}, std={teacher_feat.std().item():.4f}")

print("\n特征对比完成！")


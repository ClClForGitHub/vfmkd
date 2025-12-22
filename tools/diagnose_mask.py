#!/usr/bin/env python3
"""诊断掩码生成问题"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")

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
prompt_encoder = sam2_model.sam_prompt_encoder
mask_decoder = sam2_model.sam_mask_decoder
mask_decoder.use_high_res_features = False

# 准备prompt
point = [250, 400]
point_coords = torch.tensor([[point]], dtype=torch.float32, device=device)
point_labels = torch.tensor([[1]], dtype=torch.int32, device=device)

sparse_embeddings, dense_embeddings = prompt_encoder(
    points=(point_coords, point_labels),
    boxes=None,
    masks=None,
)

low_res_masks, iou_predictions, _, _ = mask_decoder(
    image_embeddings=adapted,
    image_pe=prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,
    repeat_image=False,
    high_res_features=None,
)

print(f"\n[INFO] 掩码生成完成")
print(f"  Low-res masks shape: {low_res_masks.shape}")
print(f"  IOU predictions: {iou_predictions[0].detach().cpu().numpy()}")

# 检查低分辨率掩码的值
print(f"\n[INFO] 低分辨率掩码统计 (before upscaling):")
for i in range(3):
    mask_i = low_res_masks[0, i]
    print(f"  Mask {i}: min={mask_i.min().item():.4f}, max={mask_i.max().item():.4f}, "
          f"mean={mask_i.mean().item():.4f}, positive_pixels={torch.sum(mask_i > 0).item()}")

# 上采样
best_idx = iou_predictions[0].argmax().item()
print(f"\n[INFO] 选择最佳掩码: Mask {best_idx}")

mask_upscaled = F.interpolate(low_res_masks[:, best_idx:best_idx+1], size=(1024, 1024), mode='bilinear', align_corners=False)
print(f"  Upscaled shape: {mask_upscaled.shape}")
print(f"  Upscaled min={mask_upscaled.min().item():.4f}, max={mask_upscaled.max().item():.4f}, "
      f"mean={mask_upscaled.mean().item():.4f}")

# 尝试不同的二值化方法
print(f"\n[INFO] 尝试不同的二值化方法:")

# 方法1: 阈值0
mask_binary1 = (mask_upscaled > 0).float()
print(f"  方法1 (>0): positive_pixels={torch.sum(mask_binary1 > 0.5).item()}")

# 方法2: sigmoid + 阈值
mask_sigmoid = torch.sigmoid(mask_upscaled)
mask_binary2 = (mask_sigmoid > 0.5).float()
print(f"  方法2 (sigmoid>0.5): positive_pixels={torch.sum(mask_binary2 > 0.5).item()}")

# 方法3: 直接使用原始值
mask_raw = mask_upscaled[0, 0].cpu().numpy()
print(f"  方法3 (raw values): positive={np.sum(mask_raw > 0)}, negative={np.sum(mask_raw < 0)}")

# 可视化所有方法
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 原图
axes[0, 0].imshow(img)
axes[0, 0].plot(point[0], point[1], 'r*', markersize=20)
axes[0, 0].set_title('Original Image with Prompt')
axes[0, 0].axis('off')

# 低分辨率掩码（未上采样）
low_res_np = low_res_masks[0, best_idx].cpu().numpy()
im1 = axes[0, 1].imshow(low_res_np, cmap='RdBu_r')
axes[0, 1].set_title(f'Low-res Mask (256x256)\nmin={low_res_np.min():.2f}, max={low_res_np.max():.2f}')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

# 上采样后的原始值
mask_up_np = mask_upscaled[0, 0].cpu().numpy()
im2 = axes[0, 2].imshow(mask_up_np, cmap='RdBu_r')
axes[0, 2].set_title(f'Upscaled Raw Values\nmin={mask_up_np.min():.2f}, max={mask_up_np.max():.2f}')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2])

# 二值化方法1
mask_b1_np = mask_binary1[0, 0].cpu().numpy()
axes[1, 0].imshow(mask_b1_np, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(f'Binary Method 1 (>0)\nWhite pixels: {np.sum(mask_b1_np > 0.5)}')
axes[1, 0].axis('off')

# 二值化方法2
mask_b2_np = mask_binary2[0, 0].cpu().numpy()
axes[1, 1].imshow(mask_b2_np, cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(f'Binary Method 2 (sigmoid>0.5)\nWhite pixels: {np.sum(mask_b2_np > 0.5)}')
axes[1, 1].axis('off')

# 叠加在原图上
overlay = np.array(img).copy()
if np.sum(mask_b2_np > 0.5) > 0:
    # 创建红色掩码
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 0] = (mask_b2_np * 255).astype(np.uint8)
    overlay = (overlay * 0.6 + mask_colored * 0.4).astype(np.uint8)
axes[1, 2].imshow(overlay)
axes[1, 2].set_title('Overlay on Original Image')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/mask_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n[INFO] 诊断可视化保存到: outputs/mask_diagnosis.png")

# 单独保存最佳二值化掩码
best_binary = mask_b2_np if np.sum(mask_b2_np) > np.sum(mask_b1_np) else mask_b1_np
mask_img = Image.fromarray((best_binary * 255).astype(np.uint8))
mask_img.save('outputs/mask_binary_only.png')
print(f"[INFO] 二值掩码保存到: outputs/mask_binary_only.png")
print(f"[INFO] 二值掩码白色像素数: {np.sum(best_binary > 0.5)}")


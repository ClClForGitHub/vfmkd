#!/usr/bin/env python3
"""创建掩码叠加到原图的可视化"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载原图
img = Image.open('datasets/coco128/images/train2017/000000000009.jpg').resize((1024, 1024))
img_arr = np.array(img)

# 加载两个掩码并resize到1024x1024
mask_repvit = Image.open('outputs/mask_RepViT_Adapter_Best.png').convert('L').resize((1024, 1024), Image.NEAREST)
mask_sam2 = Image.open('outputs/mask_SAM2_Native_Best.png').convert('L').resize((1024, 1024), Image.NEAREST)

mask_repvit_arr = np.array(mask_repvit)
mask_sam2_arr = np.array(mask_sam2)

# 创建二值掩码
mask_repvit_binary = (mask_repvit_arr > 128).astype(np.uint8)
mask_sam2_binary = (mask_sam2_arr > 128).astype(np.uint8)

# 创建叠加图像
def create_overlay(img_arr, mask_binary, color=(255, 0, 0), alpha=0.5):
    """创建掩码叠加图像"""
    overlay = img_arr.copy()
    # 在掩码区域叠加红色
    overlay[mask_binary > 0] = (
        img_arr[mask_binary > 0] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    return overlay

overlay_repvit = create_overlay(img_arr, mask_repvit_binary, color=(255, 0, 0), alpha=0.6)
overlay_sam2 = create_overlay(img_arr, mask_sam2_binary, color=(0, 255, 0), alpha=0.6)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 第一行：原图和两个掩码
axes[0, 0].imshow(img)
axes[0, 0].plot(250, 400, 'r*', markersize=30, markeredgecolor='white', markeredgewidth=2)
axes[0, 0].set_title('Original Image\nPrompt: (250, 400)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(mask_repvit_binary, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f'RepViT+Adapter Binary Mask\nWhite pixels: {np.sum(mask_repvit_binary):,} ({np.sum(mask_repvit_binary)/mask_repvit_binary.size*100:.1f}%)', 
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_sam2_binary, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'SAM2 Native Binary Mask\nWhite pixels: {np.sum(mask_sam2_binary):,} ({np.sum(mask_sam2_binary)/mask_sam2_binary.size*100:.1f}%)', 
                     fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# 第二行：叠加效果
axes[1, 0].imshow(overlay_repvit)
axes[1, 0].set_title('RepViT+Adapter Overlay\n(Red = Segmented Region)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay_sam2)
axes[1, 1].set_title('SAM2 Native Overlay\n(Green = Segmented Region)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# 计算差异
diff = np.abs(mask_repvit_binary.astype(float) - mask_sam2_binary.astype(float))
axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
axes[1, 2].set_title(f'Mask Difference\n(White = Different)\nDiff pixels: {np.sum(diff > 0):,} ({np.sum(diff)/diff.size*100:.2f}%)', 
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.suptitle('Segmentation Mask Comparison: RepViT+Adapter vs SAM2 Native\n(Binary threshold: logit > 0, equivalent to sigmoid > 0.5)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('outputs/overlay_comparison_final.png', dpi=150, bbox_inches='tight')
print("[INFO] 叠加可视化已保存到: outputs/overlay_comparison_final.png")

print("\n统计信息:")
print(f"  RepViT+Adapter 白色像素: {np.sum(mask_repvit_binary):,} ({np.sum(mask_repvit_binary)/mask_repvit_binary.size*100:.2f}%)")
print(f"  SAM2 Native 白色像素: {np.sum(mask_sam2_binary):,} ({np.sum(mask_sam2_binary)/mask_sam2_binary.size*100:.2f}%)")
print(f"  掩码差异像素: {np.sum(diff > 0):,} ({np.sum(diff)/diff.size*100:.2f}%)")
print(f"  掩码一致性: {100 - np.sum(diff)/diff.size*100:.2f}%")


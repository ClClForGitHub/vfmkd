#!/usr/bin/env python3
"""创建超清晰的可视化 - 加强对比度和标注"""
import cv2
import numpy as np

print("[INFO] 创建超清晰可视化...")

# 加载原图和掩码
img = cv2.imread('datasets/coco128/images/train2017/000000000009.jpg')
img = cv2.resize(img, (1024, 1024))

mask_repvit = cv2.imread('outputs/mask_RepViT_Adapter_Best.png', cv2.IMREAD_GRAYSCALE)
mask_repvit = cv2.resize(mask_repvit, (1024, 1024), interpolation=cv2.INTER_NEAREST)

mask_sam2 = cv2.imread('outputs/mask_SAM2_Native_Best.png', cv2.IMREAD_GRAYSCALE)
mask_sam2 = cv2.resize(mask_sam2, (1024, 1024), interpolation=cv2.INTER_NEAREST)

# 创建纯黑白版本（去除所有灰度）
mask_repvit_bw = np.zeros_like(mask_repvit)
mask_repvit_bw[mask_repvit > 128] = 255

mask_sam2_bw = np.zeros_like(mask_sam2)
mask_sam2_bw[mask_sam2 > 128] = 255

print(f"RepViT 白色像素: {np.sum(mask_repvit_bw == 255):,} ({np.sum(mask_repvit_bw == 255)/mask_repvit_bw.size*100:.1f}%)")
print(f"SAM2 白色像素: {np.sum(mask_sam2_bw == 255):,} ({np.sum(mask_sam2_bw == 255)/mask_sam2_bw.size*100:.1f}%)")

# 1. 创建超大字体的纯掩码对比
mask_comp = np.hstack([mask_repvit_bw, mask_sam2_bw])
mask_comp_rgb = cv2.cvtColor(mask_comp, cv2.COLOR_GRAY2BGR)

# 添加分割线
cv2.line(mask_comp_rgb, (1024, 0), (1024, 1024), (0, 255, 0), 5)

# 添加超大标题
cv2.putText(mask_comp_rgb, "RepViT+Adapter", (200, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)
cv2.putText(mask_comp_rgb, "SAM2 Native", (1300, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)

# 添加白色像素统计
cv2.putText(mask_comp_rgb, f"White: {np.sum(mask_repvit_bw == 255):,}", (150, 950), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4, cv2.LINE_AA)
cv2.putText(mask_comp_rgb, f"White: {np.sum(mask_sam2_bw == 255):,}", (1200, 950), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4, cv2.LINE_AA)

cv2.imwrite('outputs/SUPER_CLEAR_MASKS.png', mask_comp_rgb)
print("[SAVED] outputs/SUPER_CLEAR_MASKS.png")

# 2. 创建热力图版本（使用JET colormap）
heatmap_repvit = cv2.applyColorMap(mask_repvit, cv2.COLORMAP_JET)
heatmap_sam2 = cv2.applyColorMap(mask_sam2, cv2.COLORMAP_JET)

heatmap_comp = np.hstack([heatmap_repvit, heatmap_sam2])
cv2.line(heatmap_comp, (1024, 0), (1024, 1024), (255, 255, 255), 5)

cv2.putText(heatmap_comp, "RepViT Heatmap", (200, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
cv2.putText(heatmap_comp, "SAM2 Heatmap", (1280, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)

cv2.imwrite('outputs/SUPER_CLEAR_HEATMAP.png', heatmap_comp)
print("[SAVED] outputs/SUPER_CLEAR_HEATMAP.png")

# 3. 创建叠加版本（纯色强对比）
overlay_repvit = img.copy()
# 将掩码区域变为纯红色（不透明）
overlay_repvit[mask_repvit_bw == 255] = [0, 0, 255]

overlay_sam2 = img.copy()
# 将掩码区域变为纯绿色（不透明）
overlay_sam2[mask_sam2_bw == 255] = [0, 255, 0]

overlay_comp = np.hstack([overlay_repvit, overlay_sam2])
cv2.line(overlay_comp, (1024, 0), (1024, 1024), (255, 255, 255), 5)

cv2.putText(overlay_comp, "RepViT (RED)", (200, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
cv2.putText(overlay_comp, "SAM2 (GREEN)", (1250, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)

cv2.imwrite('outputs/SUPER_CLEAR_OVERLAY.png', overlay_comp)
print("[SAVED] outputs/SUPER_CLEAR_OVERLAY.png")

# 4. 创建单独的放大白色区域
# 找到白色像素的边界框
white_coords = np.column_stack(np.where(mask_repvit_bw == 255))
if len(white_coords) > 0:
    y_min, x_min = white_coords.min(axis=0)
    y_max, x_max = white_coords.max(axis=0)
    
    # 扩展边界
    margin = 100
    y_min = max(0, y_min - margin)
    y_max = min(1024, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(1024, x_max + margin)
    
    print(f"\n白色区域边界: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
    
    # 裁剪并放大
    crop_repvit = mask_repvit_bw[y_min:y_max, x_min:x_max]
    crop_sam2 = mask_sam2_bw[y_min:y_max, x_min:x_max]
    crop_img = img[y_min:y_max, x_min:x_max]
    
    # 放大到1024x1024
    crop_repvit_large = cv2.resize(crop_repvit, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    crop_sam2_large = cv2.resize(crop_sam2, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    crop_img_large = cv2.resize(crop_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # 拼接
    zoom_comp = np.hstack([
        cv2.cvtColor(crop_repvit_large, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(crop_sam2_large, cv2.COLOR_GRAY2BGR),
        crop_img_large
    ])
    
    cv2.putText(zoom_comp, "RepViT (ZOOMED)", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(zoom_comp, "SAM2 (ZOOMED)", (1100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(zoom_comp, "Original (ZOOMED)", (2150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
    
    cv2.imwrite('outputs/SUPER_CLEAR_ZOOMED.png', zoom_comp)
    print("[SAVED] outputs/SUPER_CLEAR_ZOOMED.png")

print("\n" + "="*60)
print("所有超清晰可视化已生成！")
print("="*60)
print("1. SUPER_CLEAR_MASKS.png - 纯黑白掩码对比")
print("2. SUPER_CLEAR_HEATMAP.png - 热力图对比")
print("3. SUPER_CLEAR_OVERLAY.png - 红绿叠加对比")
print("4. SUPER_CLEAR_ZOOMED.png - 放大白色区域")
print("\n如果还看不清，请检查图片查看器的亮度/对比度设置")


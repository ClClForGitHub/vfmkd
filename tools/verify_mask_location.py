#!/usr/bin/env python3
"""验证掩码的位置和范围"""
import cv2
import numpy as np

mask = cv2.imread('outputs/mask_RepViT_Adapter_Best.png', 0)
mask = cv2.resize(mask, (1024, 1024))

white = np.where(mask > 128)
print(f"白色像素总数: {len(white[0]):,}")

if len(white[0]) > 0:
    print(f"Y坐标范围: {white[0].min()} - {white[0].max()}")
    print(f"X坐标范围: {white[1].min()} - {white[1].max()}")
    
    center_y = int(white[0].mean())
    center_x = int(white[1].mean())
    print(f"质心位置: ({center_x}, {center_y})")
    
    print(f"\n提示点: (250, 400)")
    dist = np.sqrt((center_x - 250)**2 + (center_y - 400)**2)
    print(f"质心距提示点: {dist:.1f} 像素")
    
    # 采样一些白色像素的坐标
    print(f"\n前10个白色像素坐标 (x, y):")
    for i in range(min(10, len(white[0]))):
        print(f"  ({white[1][i]}, {white[0][i]})")


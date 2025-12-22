#!/usr/bin/env python3
"""
检查NPZ文件的内容
"""
import numpy as np
from pathlib import Path

npz_path = Path('outputs/features_v1_300/sa_1011_features.npz')

print(f"检查NPZ文件: {npz_path.name}\n")

npz_data = np.load(npz_path)

print("="*60)
print("NPZ文件包含的键:")
print("="*60)
for key in npz_data.keys():
    print(f"  - {key}")

print("\n" + "="*60)
print("各键的详细信息:")
print("="*60)

for key in npz_data.keys():
    data = npz_data[key]
    print(f"\n[{key}]")
    print(f"  shape: {data.shape}")
    print(f"  dtype: {data.dtype}")
    print(f"  mean: {data.mean():.6f}")
    print(f"  std: {data.std():.6f}")
    print(f"  min: {data.min():.6f}")
    print(f"  max: {data.max():.6f}")


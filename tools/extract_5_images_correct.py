#!/usr/bin/env python3
"""
用正确的方式提取5张图片的多尺度特征
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# 创建teacher配置
config = {
    'teacher': {
        'checkpoint_path': 'weights/sam2.1_hiera_base_plus.pt',
        'device': 'cuda',
        'model_type': 'sam2.1_hiera_b+',
        'enable_visualization': False,  # 关闭可视化以加快速度
        'feature_output_dir': 'datasets/coco128/SAM_Cache'
    }
}

from vfmkd.teachers.sam2_teacher import SAM2Teacher

print("="*80)
print("用正确方式提取5张图片的多尺度特征")
print("="*80)

# 初始化teacher
teacher = SAM2Teacher(config['teacher'])

# 选择5张图片
image_dir = Path('datasets/coco128/images/train2017')
image_files = sorted(list(image_dir.glob('*.jpg')))[:5]

print(f"\n找到 {len(image_files)} 张图片，将提取前5张:")
for img_file in image_files:
    print(f"  - {img_file.name}")

print("\n" + "="*80)
print("开始提取特征...")
print("="*80)

for idx, img_path in enumerate(image_files, 1):
    print(f"\n[{idx}/5] 处理: {img_path.name}")
    print("-"*80)
    
    # 加载图像并resize到1024x1024
    image_pil = Image.open(img_path).convert('RGB').resize((1024, 1024))
    image = np.array(image_pil)
    
    # 提取图像ID（去掉扩展名）
    image_id = img_path.stem
    
    # 提取特征
    features = teacher.extract_features(
        images=image,
        image_ids=[image_id],
        save_features=True
    )
    
    print(f"[OK] {img_path.name} 特征提取完成")

print("\n" + "="*80)
print("所有特征提取完成！")
print("="*80)

# 验证保存的NPZ文件
print("\n验证保存的NPZ文件:")
print("-"*80)
cache_dir = Path('datasets/coco128/SAM_Cache')

for img_path in image_files:
    image_id = img_path.stem
    npz_path = cache_dir / f"{image_id}_sam2_features.npz"
    
    if npz_path.exists():
        npz = np.load(npz_path)
        print(f"\n[OK] {npz_path.name}")
        print(f"   可用的特征键: {list(npz.keys())}")
        
        # 打印特征shape
        for key in npz.keys():
            if key.startswith('IMAGE_EMB') or key.startswith('FPN'):
                feat = npz[key]
                print(f"   {key}: shape={feat.shape}")
    else:
        print(f"\n[FAIL] {npz_path.name} 不存在")

print("\n" + "="*80)
print("完成！")
print("="*80)


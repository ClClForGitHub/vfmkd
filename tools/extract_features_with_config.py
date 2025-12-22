#!/usr/bin/env python3
"""
可配置的多尺度特征提取脚本
支持灵活选择保存 P3(128x128), P4(64x64), P5(32x32) 的任意组合
"""
import sys
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from vfmkd.teachers.sam2_teacher import SAM2Teacher

def main():
    parser = argparse.ArgumentParser(description='提取SAM2多尺度特征（可配置）')
    parser.add_argument('--image_dir', type=str, default='datasets/coco128/images/train2017',
                        help='图像目录路径')
    parser.add_argument('--output_dir', type=str, default='datasets/coco128/SAM_Cache',
                        help='特征输出目录')
    parser.add_argument('--num_images', type=int, default=5,
                        help='提取图像数量')
    
    # 多尺度特征开关
    parser.add_argument('--save_p3', action='store_true',
                        help='保存P3特征 (128x128, S8)')
    parser.add_argument('--save_p4', action='store_true', default=True,
                        help='保存P4特征 (64x64, S16) - 默认启用')
    parser.add_argument('--save_p5', action='store_true', default=True,
                        help='保存P5特征 (32x32, S32) - 默认启用')
    
    # 如果用户没有指定任何开关，使用默认配置（P4+P5）
    parser.add_argument('--all', action='store_true',
                        help='保存所有尺度特征 (P3+P4+P5)')
    parser.add_argument('--p4_only', action='store_true',
                        help='仅保存P4特征 (64x64)')
    parser.add_argument('--p5_only', action='store_true',
                        help='仅保存P5特征 (32x32)')
    
    args = parser.parse_args()
    
    # 处理快捷选项
    if args.all:
        args.save_p3 = True
        args.save_p4 = True
        args.save_p5 = True
    elif args.p4_only:
        args.save_p3 = False
        args.save_p4 = True
        args.save_p5 = False
    elif args.p5_only:
        args.save_p3 = False
        args.save_p4 = False
        args.save_p5 = True
    
    print("="*80)
    print("SAM2多尺度特征提取（可配置版本）")
    print("="*80)
    print(f"多尺度配置:")
    print(f"  P3 (128x128, S8):  {'[ON]' if args.save_p3 else '[OFF]'}")
    print(f"  P4 (64x64, S16):   {'[ON]' if args.save_p4 else '[OFF]'}")
    print(f"  P5 (32x32, S32):   {'[ON]' if args.save_p5 else '[OFF]'}")
    print(f"  256x256特征:       [ON] (始终保存)")
    print("="*80)
    
    # 创建teacher配置
    config = {
        'teacher': {
            'checkpoint_path': 'weights/sam2.1_hiera_base_plus.pt',
            'device': 'cuda',
            'model_type': 'sam2.1_hiera_b+',
            'enable_visualization': False,
            'feature_output_dir': args.output_dir,
            # 多尺度开关
            'save_p3': args.save_p3,
            'save_p4': args.save_p4,
            'save_p5': args.save_p5,
        }
    }
    
    # 初始化teacher
    teacher = SAM2Teacher(config['teacher'])
    
    # 获取图像列表
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob('*.jpg')))[:args.num_images]
    
    print(f"\n找到 {len(image_files)} 张图片")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    print("\n" + "="*80)
    print("开始提取特征...")
    print("="*80)
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {img_path.name}")
        print("-"*80)
        
        # 加载图像并resize到1024x1024
        image_pil = Image.open(img_path).convert('RGB').resize((1024, 1024))
        image = np.array(image_pil)
        
        # 提取图像ID
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
    cache_dir = Path(args.output_dir)
    
    for img_path in image_files:
        image_id = img_path.stem
        npz_path = cache_dir / f"{image_id}_sam2_features.npz"
        
        if npz_path.exists():
            npz = np.load(npz_path)
            print(f"\n[OK] {npz_path.name}")
            
            # 统计文件大小
            file_size_mb = npz_path.stat().st_size / (1024 * 1024)
            print(f"   文件大小: {file_size_mb:.1f} MB")
            
            # 列出特征键
            feature_keys = [k for k in npz.keys() if k.startswith('IMAGE_EMB') or k.startswith('P')]
            print(f"   特征键: {feature_keys}")
            
            # 打印特征shape
            for key in feature_keys:
                feat = npz[key]
                print(f"   {key}: shape={feat.shape}")
        else:
            print(f"\n[FAIL] {npz_path.name} 不存在")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == '__main__':
    main()


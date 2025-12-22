#!/usr/bin/env python3
"""
第一版特征提取脚本
使用SAM2.1hiera教师模型提取16x下采样256通道特征
同时从SA-1B JSON生成边缘图(64x64和256x256)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import cv2
from pycocotools import mask as mask_utils
import torch.nn.functional as F

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.teachers.sam2_teacher import SAM2Teacher


class SA1BFeatureExtractor:
    """SA-1B特征提取器"""
    
    def __init__(self, config):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建SAM2教师模型
        self.teacher = SAM2Teacher(config['teacher'])
        
        # 边缘提取配置
        self.kernel_size = config.get('kernel_size', 3)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        
        print(f"✅ 特征提取器初始化完成")
        print(f"设备: {self.device}")
        print(f"教师模型: {self.teacher.model_name}")
    
    def extract_edges_and_weights_optimized(self, json_path, edge_sizes=[256, 64, 32], weight_sizes=[128, 64, 32]):
        """
        优化版：合并解码 + 只做一次形态学操作 + 同时生成边缘图和权重图
        
        Args:
            json_path: JSON标注文件路径
            edge_sizes: 边缘图目标尺寸列表
            weight_sizes: 权重图目标尺寸列表
            
        Returns:
            (edge_maps, weight_maps): 边缘图字典和权重图字典
        """
        # 加载JSON标注文件
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        image_info = data['image']
        height = image_info['height']
        width = image_info['width']
        
        # 获取所有RLE标注
        annotations = data['annotations']
        
        if len(annotations) == 0:
            # 没有标注，返回空图
            union_mask = np.zeros((height, width), dtype=np.uint8)
        else:
            # 单进程顺序解码并合并（避免Windows多进程开销）
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in annotations:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                union_mask = np.maximum(union_mask, mask)
        
        # 在合并后的掩码上提取边缘（只做一次形态学操作！）
        combined_edge_map = cv2.morphologyEx(union_mask, cv2.MORPH_GRADIENT, self.kernel)
        
        # === 生成多尺度边缘图 ===
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        # === 生成权重图（复用union_mask） ===
        weight_maps = {}
        for size in weight_sizes:
            # 下采样到特征图分辨率
            union_tensor = torch.from_numpy(union_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            fg_prob = F.adaptive_avg_pool2d(union_tensor, (size, size)).squeeze().numpy()
            
            # 前景权重：area归一化
            fg_binary = (fg_prob > 0.5).astype(np.float32)
            num_fg = np.clip(fg_binary.sum(), a_min=1, a_max=None)
            fg_map = fg_binary / num_fg
            
            # 背景权重：归一化
            bg_map = 1.0 - fg_binary
            bg_sum = bg_map.sum()
            if bg_sum > 0:
                bg_map = bg_map / bg_sum
            
            weight_maps[size] = {
                'fg_map': fg_map.astype(np.float32),
                'bg_map': bg_map.astype(np.float32)
            }
        
        return edge_maps, weight_maps
    
    def process_single_image(self, image_path, json_path, output_dir):
        """
        处理单张图像：提取特征和边缘图
        
        Args:
            image_path: 图像文件路径
            json_path: JSON标注文件路径
            output_dir: 输出目录
            
        Returns:
            dict: 处理结果
        """
        import time
        image_id = Path(image_path).stem
        timing = {}  # 记录各步骤耗时
        
        try:
            # 1. 加载图像
            t0 = time.time()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            timing['load_image'] = time.time() - t0
            
            # 2. 提取SAM2特征（SAM2 teacher内部使用官方transform：resize + normalize）
            t0 = time.time()
            features = self.teacher.extract_features(
                image_rgb, 
                image_ids=[image_id], 
                save_features=False  # 我们手动保存
            )
            timing['sam2_features'] = time.time() - t0
            
            # 3. 同时提取边缘图和权重图（优化版：合并解码，只做一次形态学）
            t0 = time.time()
            edge_maps, weight_maps = self.extract_edges_and_weights_optimized(
                json_path, 
                edge_sizes=[256, 64, 32],  # 移除128以加速
                weight_sizes=[128, 64, 32]
            )
            timing['edges_and_weights'] = time.time() - t0
            
            # 5. 保存特征、边缘图和权重图
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存NPZ文件（特征 + 边缘图 + 权重图）
            save_data = {}
            
            # 添加SAM2特征
            for key, feat in features.items():
                save_data[key] = feat.detach().cpu().numpy()
            
            # 添加边缘图（多尺度，移除128x128以加速）
            save_data['edge_original'] = edge_maps['original']  # 原图尺寸边缘图
            save_data['edge_256x256'] = edge_maps[256]         # 256x256边缘图
            save_data['edge_64x64'] = edge_maps[64]            # 64x64边缘图 (对应s16, P4)
            save_data['edge_32x32'] = edge_maps[32]            # 32x32边缘图 (对应s32, P5)
            
            # 添加前景/背景权重图（多尺度）
            for size in [128, 64, 32]:
                save_data[f'fg_map_{size}x{size}'] = weight_maps[size]['fg_map']
                save_data[f'bg_map_{size}x{size}'] = weight_maps[size]['bg_map']
            
            # 添加元数据
            save_data['image_id'] = image_id
            save_data['image_shape'] = np.array(image_rgb.shape)
            save_data['model_type'] = self.teacher.model_type
            
            # 保存NPZ文件
            t0 = time.time()
            npz_file = output_path / f"{image_id}_features.npz"
            np.savez(npz_file, **save_data)
            timing['save_npz'] = time.time() - t0
            
            # 计算总时间
            total_time = sum(timing.values())
            
            return {
                'success': True,
                'image_id': image_id,
                'npz_file': npz_file,
                'feature_shape': features['P4_S16'].shape,  # 使用P4_S16替代IMAGE_EMB_S16
                'edge_shapes': {f'edge_{k}': v.shape for k, v in edge_maps.items()},
                'timing': timing,
                'total_time': total_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'image_id': image_id,
                'error': str(e)
            }
    
    def batch_extract_features(self, data_dir, output_dir, max_images=None):
        """
        批量提取特征和边缘图
        
        Args:
            data_dir: SA-1B数据目录
            output_dir: 输出目录
            max_images: 最大处理图像数量
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        # 获取所有图像文件
        image_files = list(data_path.glob("*.jpg"))
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"开始处理 {len(image_files)} 个图像...")
        
        success_count = 0
        error_count = 0
        avg_timing = {'load_image': 0, 'sam2_features': 0, 'edges_and_weights': 0, 'save_npz': 0}
        
        for image_file in tqdm(image_files, desc="提取特征和边缘图"):
            # 查找对应的JSON文件
            json_file = data_path / f"{image_file.stem}.json"
            if not json_file.exists():
                print(f"⚠️  跳过 {image_file.stem}: 找不到对应的JSON文件")
                error_count += 1
                continue
            
            # 处理图像
            result = self.process_single_image(image_file, json_file, output_path)
            
            if result['success']:
                success_count += 1
                # 累加计时统计
                for key in avg_timing:
                    if key in result.get('timing', {}):
                        avg_timing[key] += result['timing'][key]
                
                # 每10张打印一次详细计时
                if success_count % 10 == 0:
                    print(f"\n✅ {result['image_id']}: 总{result['total_time']:.2f}s "
                          f"[加载{result['timing']['load_image']:.3f}s | "
                          f"SAM2特征{result['timing']['sam2_features']:.3f}s | "
                          f"边缘+权重图{result['timing']['edges_and_weights']:.3f}s | "
                          f"保存{result['timing']['save_npz']:.3f}s]")
            else:
                error_count += 1
                print(f"❌ {result['image_id']}: {result['error']}")
        
        print(f"\n🎉 特征提取完成!")
        print(f"成功: {success_count} 个")
        print(f"失败: {error_count} 个")
        print(f"输出目录: {output_path}")
        
        # 打印平均耗时统计
        if success_count > 0:
            print(f"\n⏱️  平均耗时 (每张):")
            for key, total_time in avg_timing.items():
                avg_time = total_time / success_count
                print(f"  {key}: {avg_time:.3f}s")
            total_avg = sum(avg_timing.values()) / success_count
            print(f"  总计: {total_avg:.3f}s")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="第一版特征提取脚本")
    parser.add_argument("--data-dir", type=str, required=True, help="SA-1B数据目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max-images", type=int, default=None, help="最大处理图像数量")
    parser.add_argument("--teacher-model", type=str, default="sam2.1_hiera_b+", 
                       choices=["sam2.1_hiera_t", "sam2.1_hiera_s", "sam2.1_hiera_b+", "sam2.1_hiera_l"],
                       help="教师模型类型")
    parser.add_argument("--checkpoint", type=str, default=None, help="权重文件路径")
    parser.add_argument("--kernel-size", type=int, default=3, help="边缘提取核大小")
    parser.add_argument("--diag-compare", action='store_true', help="启用诊断：保存前对比分布，打印mean/std")
    parser.add_argument("--diag-fallback", action='store_true', help="启用回退：若std异常则回退为/255实时特征")
    
    args = parser.parse_args()
    
    # 构建配置
    # 统一权重路径：b+ 对应 base_plus 权重文件名
    _ckpt = args.checkpoint
    if _ckpt is None:
        if args.teacher_model.endswith('hiera_b+'):
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'
        else:
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'

    config = {
        'teacher': {
            'model_type': args.teacher_model,
            'checkpoint_path': _ckpt,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enable_visualization': False,  # 关闭可视化以提高速度
            'feature_output_dir': args.output_dir,
            'enable_diag_compare': bool(args.diag_compare),
            'fallback_if_high_std': bool(args.diag_fallback),
        },
        'kernel_size': args.kernel_size
    }
    
    print("=== 第一版特征提取 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"教师模型: {args.teacher_model}")
    print(f"权重文件: {config['teacher']['checkpoint_path']}")
    print(f"最大图像数: {args.max_images or '全部'}")
    
    # 创建特征提取器
    extractor = SA1BFeatureExtractor(config)
    
    # 批量提取特征
    extractor.batch_extract_features(
        args.data_dir,
        args.output_dir,
        args.max_images
    )


if __name__ == "__main__":
    main()

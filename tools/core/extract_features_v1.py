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

# 添加项目路径和SAM2路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sam2_path = project_root / "vfmkd" / "sam2"
if str(sam2_path) not in sys.path:
    sys.path.insert(0, str(sam2_path))

from vfmkd.teachers.sam2_teacher import SAM2Teacher
from tools.core.bbox.test_bbox_strategies import (
    load_sa_json,
    compute_strategy_geometry_color,
)


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
        
        # 选框策略配置                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             说
        self.max_instances = config.get('max_instances', 1)
        self.enable_bbox_selection = config.get('enable_bbox_selection', True)
        
        print(f"✅ 特征提取器初始化完成")
        print(f"设备: {self.device}")
        print(f"教师模型: {self.teacher.model_name}")
    
    def extract_edges_and_weights_optimized(self, json_path, edge_sizes=[256, 64, 32], weight_sizes=[128, 64, 32]):
        """
        优化版：使用Method B（每实例提取边缘后合并）+ 同时生成边缘图和权重图
        完全复刻edge_comparison中的Method B（CPU优化版本）
        
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
        
        # === 使用Method B提取边缘（CPU优化版本，与edge_comparison完全一致）===
        if len(annotations) == 0:
            # 没有标注，返回空图
            union_mask = np.zeros((height, width), dtype=np.uint8)
            combined_edge_map = np.zeros((height, width), dtype=np.uint8)
        else:
            # Method B：每个实例单独提取边缘后合并
            combined_edge_map = np.zeros((height, width), dtype=np.uint8)
            union_mask = np.zeros((height, width), dtype=np.uint8)
            
            for ann in annotations:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)  # 从RLE解码
                
                # 合并掩码（用于权重图）
                union_mask = np.maximum(union_mask, mask)
        
                # 对每个实例单独提取边缘
                edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
                # 二值化并确保uint8类型（避免类型不匹配和溢出警告）
                edge = (edge > 0).astype(np.uint8)
                
                # 使用bitwise_or替代logical_or（直接在uint8上操作）
                combined_edge_map = np.bitwise_or(combined_edge_map, edge)
        
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
            
            # 4. 应用geometry_color策略选择框和掩码（新增，复用已有image_rgb）
            bbox_result = None
            if self.enable_bbox_selection:
                t0 = time.time()
                try:
                    # 准备图像信息（复用已有的image_rgb，避免重复读取）
                    H, W = image_rgb.shape[:2]
                    data = {
                        'image': {
                            'height': H,
                            'width': W,
                            'h': H,
                            'w': W,
                        }
                    }
                    
                    # 加载JSON标注（复用json_path）
                    sa_data = load_sa_json(str(json_path))
                    annotations = sa_data.get('annotations', [])
                    
                    # 应用策略（直接使用已有的image_rgb，注意需要BGR格式）
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    selected_components = compute_strategy_geometry_color(
                        data=data,
                        annotations=annotations,
                        image_rgb=image_bgr,  # 函数内部会转换，这里传入BGR
                        clip_data=None,
                        max_instances=self.max_instances,
                        max_display=10,
                        debug_trace=None,
                    )
                    
                    # 提取框和掩码
                    if len(selected_components) > 0:
                        bboxes = []
                        masks = []
                        for comp in selected_components:
                            bboxes.append(comp['box'])  # [x, y, w, h]
                            masks.append(comp['mask'])  # [H, W] uint8
                        
                        bbox_result = {
                            'has_bbox': True,
                            'bboxes': np.array(bboxes, dtype=np.float32),
                            'masks': masks,
                        }
                    else:
                        bbox_result = {
                            'has_bbox': False,
                            'bboxes': np.empty((0, 4), dtype=np.float32),
                            'masks': [],
                        }
                    timing['bbox_selection'] = time.time() - t0
                except Exception as e:
                    # 选框失败不影响其他数据保存，使用空结果
                    print(f"⚠️  选框策略失败 {image_id}: {e}")
                    bbox_result = {
                        'has_bbox': False,
                        'bboxes': np.empty((0, 4), dtype=np.float32),
                        'masks': [],
                    }
                    timing['bbox_selection'] = 0.0
            else:
                bbox_result = None
                timing['bbox_selection'] = 0.0
            
            # 5. 保存特征、边缘图和权重图
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存NPZ文件（特征 + 边缘图 + 权重图）
            save_data = {}
            
            # 添加SAM2特征
            for key, feat in features.items():
                save_data[key] = feat.detach().cpu().numpy()
            
            # 添加边缘图（多尺度，移除edge_original以节省空间）
            # save_data['edge_original'] = edge_maps['original']  # 原图尺寸边缘图（已移除以节省空间）
            save_data['edge_256x256'] = edge_maps[256]         # 256x256边缘图
            save_data['edge_64x64'] = edge_maps[64]            # 64x64边缘图 (对应s16, P4)
            save_data['edge_32x32'] = edge_maps[32]            # 32x32边缘图 (对应s32, P5)
            
            # 添加前景/背景权重图（多尺度）
            for size in [128, 64, 32]:
                save_data[f'fg_map_{size}x{size}'] = weight_maps[size]['fg_map']
                save_data[f'bg_map_{size}x{size}'] = weight_maps[size]['bg_map']
            
            # 添加选框结果（如果启用）
            if bbox_result is not None:
                save_data['has_bbox'] = np.array(bbox_result['has_bbox'], dtype=bool)
                save_data['num_bboxes'] = np.array(len(bbox_result['bboxes']), dtype=np.int32)
                save_data['bboxes'] = bbox_result['bboxes']  # (N, 4) or (0, 4)
                if bbox_result['has_bbox'] and len(bbox_result['masks']) > 0:
                    save_data['masks'] = np.array(bbox_result['masks'], dtype=object)
                save_data['geometry_color_flag'] = np.array(1, dtype=np.uint8)  # 标记已处理
            
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
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(data_path.glob("*.jpg"))
        if max_images:
            image_files = image_files[:max_images]
        
        # 检查已存在的NPZ文件（去重）
        existing_npz = set()
        for npz_file in output_path.glob("*_features.npz"):
            image_id = npz_file.stem.replace('_features', '')
            existing_npz.add(image_id)
        
        print(f"📁 总图像数: {len(image_files)}")
        print(f"✅ 已提取: {len(existing_npz)} 个")
        print(f"⏳ 待处理: {len(image_files) - len(existing_npz)} 个")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        avg_timing = {'load_image': 0, 'sam2_features': 0, 'edges_and_weights': 0, 
                      'bbox_selection': 0, 'save_npz': 0}
        
        for image_file in tqdm(image_files, desc="提取特征和边缘图"):
            image_id = image_file.stem
            
            # 去重检查：如果NPZ文件已存在，跳过
            if image_id in existing_npz:
                skipped_count += 1
                continue
            
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
                    bbox_time = result['timing'].get('bbox_selection', 0)
                    print(f"\n✅ {result['image_id']}: 总{result['total_time']:.2f}s "
                          f"[加载{result['timing']['load_image']:.3f}s | "
                          f"SAM2特征{result['timing']['sam2_features']:.3f}s | "
                          f"边缘+权重图{result['timing']['edges_and_weights']:.3f}s | "
                          f"选框{bbox_time:.3f}s | "
                          f"保存{result['timing']['save_npz']:.3f}s]")
            else:
                error_count += 1
                print(f"❌ {result['image_id']}: {result['error']}")
        
        print(f"\n🎉 特征提取完成!")
        print(f"✅ 本次成功: {success_count} 个")
        print(f"⏭️  已跳过(去重): {skipped_count} 个")
        print(f"❌ 失败: {error_count} 个")
        print(f"📊 总计已提取: {len(existing_npz) + success_count} 个")
        print(f"📁 输出目录: {output_path}")
        
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
    parser.add_argument("--device", type=str, default="cuda:6", help="指定GPU设备 (如: cuda:0, cuda:3)，默认cuda:6")
    parser.add_argument("--diag-compare", action='store_true', help="启用诊断：保存前对比分布，打印mean/std")
    parser.add_argument("--diag-fallback", action='store_true', help="启用回退：若std异常则回退为/255实时特征")
    parser.add_argument('--max-instances', type=int, default=1,
                       help='选框策略最多选择的实例数（默认1）')
    parser.add_argument('--enable-bbox-selection', action='store_true', default=True,
                       help='启用选框策略（默认启用）')
    parser.add_argument('--disable-bbox-selection', dest='enable_bbox_selection', action='store_false',
                       help='禁用选框策略')
    
    args = parser.parse_args()
    
    # 构建配置
    # 统一权重路径：b+ 对应 base_plus 权重文件名
    _ckpt = args.checkpoint
    if _ckpt is None:
        if args.teacher_model.endswith('hiera_b+'):
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'
        else:
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'
    
    # 确定使用的设备
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'teacher': {
            'model_type': args.teacher_model,
            'checkpoint_path': _ckpt,
            'device': device,
            'enable_visualization': False,  # 关闭可视化以提高速度
            'feature_output_dir': args.output_dir,
            'enable_diag_compare': bool(args.diag_compare),
            'fallback_if_high_std': bool(args.diag_fallback),
        },
        'kernel_size': args.kernel_size,
        'max_instances': args.max_instances,
        'enable_bbox_selection': args.enable_bbox_selection,
    }
    
    print("=== 第一版特征提取 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"教师模型: {args.teacher_model}")
    print(f"权重文件: {config['teacher']['checkpoint_path']}")
    print(f"计算设备: {device}")
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

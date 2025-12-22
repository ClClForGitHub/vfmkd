#!/usr/bin/env python3
"""
SA-1B边缘提取工具
从SA-1B数据集的JSON标注文件中提取边缘图，用于蒸馏训练的监督信号
"""

import json
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils


class SA1BEdgeExtractor:
    """SA-1B边缘提取器"""
    
    def __init__(self, kernel_size=3):
        """
        初始化边缘提取器
        
        Args:
            kernel_size: 形态学操作的核大小
        """
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    def extract_edges_from_json(self, json_path, output_dir=None, save_formats=['npy'], target_size=256):
        """
        从JSON文件提取边缘图
        
        Args:
            json_path: JSON标注文件路径
            output_dir: 输出目录
            save_formats: 保存格式列表 ['npy']
            target_size: 目标尺寸（用于训练）
            
        Returns:
            tuple: (原图边缘, 256x256边缘)
        """
        # 加载JSON标注文件
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        image_info = data['image']
        height = image_info['height']
        width = image_info['width']
        
        # 创建空的边缘图
        combined_edge_map = np.zeros((height, width), dtype=np.uint8)
        
        # 遍历所有分割标注
        annotations = data['annotations']
        print(f"图像中总共有 {len(annotations)} 个分割掩码")
        
        for ann in annotations:
            # 解码RLE掩码
            rle = ann['segmentation']
            binary_mask = mask_utils.decode(rle).astype(np.uint8)
            
            # 提取边缘（形态学梯度）
            gradient = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, self.kernel)
            
            # 合并到总边缘图
            combined_edge_map = np.logical_or(combined_edge_map, gradient)
        
        # 原图尺寸边缘图（二值）
        original_edge_map = combined_edge_map.astype(np.uint8)
        
        # 生成目标尺寸边缘图 - 使用INTER_AREA插值
        # 先转换为float [0, 1]
        combined_edge_map_float = combined_edge_map.astype(np.float32)
        
        # 使用INTER_AREA插值下采样
        downsampled_gray = cv2.resize(
            combined_edge_map_float, 
            (target_size, target_size),  # cv2.resize是(宽, 高)
            interpolation=cv2.INTER_AREA
        )
        
        # 阈值化：只要 > 0 就设为 1
        resized_edge_map = (downsampled_gray > 0).astype(np.uint8)
        
        # 保存文件
        if output_dir and save_formats:
            json_file = Path(json_path)
            base_name = json_file.stem
            
            for fmt in save_formats:
                if fmt == 'npy':
                    # 保存原图尺寸numpy格式
                    npy_orig_path = Path(output_dir) / f"{base_name}_edges_original.npy"
                    np.save(npy_orig_path, original_edge_map)
                    
                    # 保存256x256 numpy格式（用于训练）
                    npy_resized_path = Path(output_dir) / f"{base_name}_edges_{target_size}x{target_size}.npy"
                    np.save(npy_resized_path, resized_edge_map)
                    
                    print(f"边缘图已保存: {base_name}_edges_original.npy, {base_name}_edges_{target_size}x{target_size}.npy")
        
        return original_edge_map, resized_edge_map
    
    def batch_extract_edges(self, data_dir, output_dir, num_images=None, target_size=256, save_formats=['npy']):
        """
        批量提取边缘图
        
        Args:
            data_dir: SA-1B数据目录
            output_dir: 输出目录
            num_images: 处理的图片数量（None表示处理所有）
            target_size: 目标尺寸
            save_formats: 保存格式列表
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有JSON文件
        json_files = list(data_path.glob("*.json"))
        if num_images:
            json_files = json_files[:num_images]
        
        print(f"开始处理 {len(json_files)} 个文件...")
        print(f"目标尺寸: {target_size}x{target_size}")
        print(f"保存格式: {save_formats}")
        
        for json_file in tqdm(json_files, desc="提取边缘图"):
            try:
                # 提取边缘图
                original_edges, resized_edges = self.extract_edges_from_json(
                    json_file, 
                    output_path, 
                    save_formats=save_formats,
                    target_size=target_size
                )
                
                # 打印统计信息
                print(f"  {json_file.stem}: 原图{original_edges.shape} -> 256x256{resized_edges.shape}")
                print(f"    原图边缘像素: {np.sum(original_edges > 0)}")
                print(f"    256x256边缘像素: {np.sum(resized_edges > 0)}")
                
            except Exception as e:
                print(f"处理 {json_file} 时出错: {e}")
                continue
        
        print(f"边缘提取完成！共处理 {len(json_files)} 个文件")
        print(f"输出目录: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SA-1B边缘提取工具")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="SA-1B数据目录路径"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="边缘图输出目录"
    )
    parser.add_argument(
        "--num-images", 
        type=int, 
        default=None,
        help="处理的图片数量（默认处理所有）"
    )
    parser.add_argument(
        "--kernel-size", 
        type=int, 
        default=3,
        help="形态学操作核大小（默认3）"
    )
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=256,
        help="目标尺寸（默认256）"
    )
    parser.add_argument(
        "--save-formats", 
        nargs='+', 
        default=['npy'],
        choices=['npy'],
        help="保存格式（默认npy）"
    )
    parser.add_argument(
        "--single-file", 
        type=str, 
        default=None,
        help="处理单个JSON文件（用于测试）"
    )
    
    args = parser.parse_args()
    
    # 创建边缘提取器
    extractor = SA1BEdgeExtractor(kernel_size=args.kernel_size)
    
    if args.single_file:
        # 处理单个文件
        print(f"处理单个文件: {args.single_file}")
        original_edges, resized_edges = extractor.extract_edges_from_json(
            args.single_file, 
            args.output_dir,
            save_formats=args.save_formats,
            target_size=args.target_size
        )
        print(f"原图边缘图形状: {original_edges.shape}")
        print(f"256x256边缘图形状: {resized_edges.shape}")
        print(f"原图边缘像素数量: {np.sum(original_edges > 0)}")
        print(f"256x256边缘像素数量: {np.sum(resized_edges > 0)}")
    else:
        # 批量处理
        extractor.batch_extract_edges(
            args.data_dir, 
            args.output_dir, 
            args.num_images,
            args.target_size,
            args.save_formats
        )


if __name__ == "__main__":
    main()

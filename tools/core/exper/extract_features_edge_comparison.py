#!/usr/bin/env python3
"""
边缘提取方法对比实验脚本
对比三种边缘提取方法：
- Method A (Baseline): Union mask then extract edge
- Method B (Improvement 1): Extract edge per instance then merge
- Method C (Improvement 2): Instance mask map (different values) then morphology

只关注边缘提取速度，三种方法都从同一个JSON开始独立处理
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import cv2
from pycocotools import mask as mask_utils
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any

# 添加项目路径和SAM2路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def set_cuda_device(device_str):
    """
    Set CUDA device and ensure single GPU usage
    
    Args:
        device_str: Device string like 'cuda:5' or 'cuda'
    """
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Extract device index from string like 'cuda:5'
    if ':' in device_str:
        device_idx = int(device_str.split(':')[1])
    else:
        device_idx = 0
    
    # Check if device exists
    if device_idx >= torch.cuda.device_count():
        print(f"[WARNING] Device cuda:{device_idx} does not exist. Available devices: {torch.cuda.device_count()}")
        print(f"[WARNING] Using default device: cuda:0")
        device_idx = 0
    
    # Set CUDA device (this ensures all operations use this device)
    torch.cuda.set_device(device_idx)
    device = torch.device(f'cuda:{device_idx}')
    
    print(f"[CUDA] Using device: cuda:{device_idx} ({torch.cuda.get_device_name(device_idx)})")
    print(f"[CUDA] Current device: {torch.cuda.current_device()}")
    
    # Verify device setting
    test_tensor = torch.randn(1, 1).to(device)
    if test_tensor.device.index != device_idx:
        raise RuntimeError(f"Device setting failed: tensor on {test_tensor.device} but expected cuda:{device_idx}")
    
    return device


# 导入 bbox 提取器
from tools.core.bbox import SA1BInstanceBoxExtractor


class EdgeExtractionComparison:
    """边缘提取方法对比实验"""
    
    def __init__(self, config):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 边缘提取配置
        self.kernel_size = config.get('kernel_size', 3)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        
        # CUDA设备配置
        self.device = None
        if 'device' in config:
            device_str = config['device']
            if isinstance(device_str, str) and 'cuda' in device_str:
                if ':' in device_str:
                    device_idx = int(device_str.split(':')[1])
                else:
                    device_idx = 0
                self.device = torch.device(f'cuda:{device_idx}')
            elif hasattr(device_str, 'index'):
                self.device = device_str
            else:
                self.device = torch.device('cpu')
        
        # 创建PyTorch形态学梯度kernel（用于CUDA加速）
        if self.device and self.device.type == 'cuda':
            # 形态学梯度 = dilation - erosion
            # dilation用max_pool2d实现，erosion用min_pool2d实现
            self.use_cuda = True
            self.kernel_tensor = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=torch.float32, device=self.device)
        else:
            self.use_cuda = False
            self.kernel_tensor = None
        
        print(f"✅ Edge Extraction Comparison Initialized")
        print(f"Kernel size: {self.kernel_size}")
        print(f"CUDA acceleration: {self.use_cuda}")
        if self.use_cuda:
            print(f"Device: {self.device}")
    
    def method_a_baseline(self, json_path, edge_sizes=[256, 64, 32]):
        """
        Method A (Baseline): Union mask then extract edge
        从JSON开始独立处理：解码→合并掩码→提取边缘
        """
        # 从JSON开始处理（独立）
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height = data['image']['height']
        width = data['image']['width']
        annotations = data['annotations']
        
        # 解码并合并掩码
        if len(annotations) == 0:
            union_mask = np.zeros((height, width), dtype=np.uint8)
        else:
            union_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in annotations:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)  # 从RLE解码
                union_mask = np.maximum(union_mask, mask)
        
        # 提取边缘
        combined_edge_map = cv2.morphologyEx(union_mask, cv2.MORPH_GRADIENT, self.kernel)
        
        # 生成多尺度边缘图
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        return edge_maps, union_mask
    def _morphological_gradient_cuda(self, mask_tensor):
        """
        使用CUDA实现的形态学梯度：dilation - erosion
        mask_tensor: [N, 1, H, W] float32 tensor on GPU
        Returns: [N, 1, H, W] gradient tensor
        """
        # Dilation: max pooling (取邻域最大值)
        dilation = F.max_pool2d(
            mask_tensor,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
        
        # Erosion: 使用技巧 -max_pool2d(-mask)
        erosion = -F.max_pool2d(
            -mask_tensor,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
        
        # Gradient = dilation - erosion
        gradient = dilation - erosion
        
        return gradient

    
    
    
    def method_b_per_instance(self, json_path, edge_sizes=[256, 64, 32], use_cuda=False):
        """
        Method B (Optimized): Extract edge per instance then merge
        优化：使用bitwise_or替代logical_or，去掉循环内的类型转换
        CUDA加速：如果use_cuda=True且设备可用，批量处理mask在GPU上
        """
        # 从JSON开始处理（独立）
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height = data['image']['height']
        width = data['image']['width']
        annotations = data['annotations']
        
        # 决定是否使用CUDA
        use_cuda_accel = use_cuda and self.use_cuda and self.device is not None
        
        if use_cuda_accel and len(annotations) > 0:
            # CUDA加速版本：分批处理mask避免OOM
            mask_list = []
            union_mask = np.zeros((height, width), dtype=np.uint8)
            
            for ann in annotations:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle).astype(np.float32)
                union_mask = np.maximum(union_mask, mask.astype(np.uint8))
                mask_list.append(mask)
            
            # 分批处理：每批最多32个mask（可根据GPU内存调整）
            batch_size = 32
            combined_edge_map = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(0, len(mask_list), batch_size):
                batch_masks = mask_list[i:i+batch_size]
                
                # 堆叠成tensor [N, 1, H, W]
                masks_tensor = torch.stack([
                    torch.from_numpy(m).to(self.device).unsqueeze(0)
                    for m in batch_masks
                ])  # [N, 1, H, W]
                
                # 批量计算形态学梯度
                edges_tensor = self._morphological_gradient_cuda(masks_tensor)  # [N, 1, H, W]
                
                # 合并边缘：使用torch.max合并当前批次
                batch_edges = edges_tensor.max(dim=0)[0]  # [1, H, W]
                batch_edges = (batch_edges > 0).float()
                
                # 转回CPU numpy并合并到总结果
                batch_edge_map = (batch_edges.squeeze().cpu().numpy() > 0).astype(np.uint8)
                combined_edge_map = np.bitwise_or(combined_edge_map, batch_edge_map)
                
                # 清理GPU内存
                del masks_tensor, edges_tensor, batch_edges
                torch.cuda.empty_cache()
        else:
            # CPU版本（原始优化版本）
            combined_edge_map = np.zeros((height, width), dtype=np.uint8)
            union_mask = np.zeros((height, width), dtype=np.uint8)
            
            if len(annotations) == 0:
                pass
            else:
                for ann in annotations:
                    rle = ann['segmentation']
                    mask = mask_utils.decode(rle)  # 从RLE解码
                    
                    # 合并掩码（用于返回）
                    union_mask = np.maximum(union_mask, mask)
                    
                    # 对每个实例单独提取边缘
                    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
                    # 二值化并确保uint8类型（避免类型不匹配和溢出警告）
                    edge = (edge > 0).astype(np.uint8)
                    
                    # 使用bitwise_or替代logical_or（直接在uint8上操作）
                    combined_edge_map = np.bitwise_or(combined_edge_map, edge)
        
        # 生成多尺度边缘图
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        return edge_maps, union_mask
    def method_b_per_instance_original(self, json_path, edge_sizes=[256, 64, 32]):
        """
        Method B (Original): Extract edge per instance then merge
        原始版本：使用logical_or + astype(np.uint8)（每次循环都转换）
        用于对比优化前后的效果
        """
        # 从JSON开始处理（独立）
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height = data['image']['height']
        width = data['image']['width']
        annotations = data['annotations']
        
        # 合并边缘图
        combined_edge_map = np.zeros((height, width), dtype=np.uint8)
        union_mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(annotations) == 0:
            pass
        else:
            for ann in annotations:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)  # 从RLE解码
                
                # 合并掩码（用于返回）
                union_mask = np.maximum(union_mask, mask)
                
                # 对每个实例单独提取边缘
                edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
                
                # 原始版本：使用logical_or + 每次循环都转换类型
                combined_edge_map = np.logical_or(combined_edge_map, edge).astype(np.uint8)
        
        # 生成多尺度边缘图
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        return edge_maps, union_mask
    
    def method_c_instance_mask(self, json_path, edge_sizes=[256, 64, 32]):
        """
        Method C (Improvement 2): Instance mask map (different values) then morphology
        从JSON开始独立处理：解码→实例掩码图（不同值）→形态学操作
        优化：直接使用uint8，无需检查
        """
        # 从JSON开始处理（独立）
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height = data['image']['height']
        width = data['image']['width']
        annotations = data['annotations']
        
        # 实例掩码图：每个实例使用不同的值（直接使用uint8，1, 2, 3, ...）
        instance_mask_map = np.zeros((height, width), dtype=np.uint8)
        union_mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(annotations) == 0:
            pass
        else:
            for idx, ann in enumerate(annotations):
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)  # 从RLE解码
                
                # 合并掩码（用于返回普通union_mask）
                union_mask = np.maximum(union_mask, mask)
                
                # 直接分配：第一个实例=1，第二个=2，...（无需检查，假设<256个实例）
                instance_mask_map[mask > 0] = idx + 1
        
        # 直接应用形态学梯度（无需类型转换和检查）
        combined_edge_map = cv2.morphologyEx(instance_mask_map, cv2.MORPH_GRADIENT, self.kernel)
        
        # 转换为二值：任何值变化（非零梯度）都是边缘
        combined_edge_map = (combined_edge_map > 0).astype(np.uint8)
        
        # 生成多尺度边缘图
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        # 返回edge_maps, union_mask, instance_mask_map（用于可视化）
        return edge_maps, union_mask, instance_mask_map
    
    def method_c_instance_mask_optimized(self, json_path, edge_sizes=[256, 64, 32]):
        """
        Method C (Optimized): 质量优化版本 - 处理重叠区域
        非重叠区域：使用实例掩码图一次形态学
        重叠区域：单独提取边缘
        """
        # 从JSON开始处理（独立）
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height = data['image']['height']
        width = data['image']['width']
        annotations = data['annotations']
        
        # 实例掩码图：每个实例使用不同的值
        instance_mask_map = np.zeros((height, width), dtype=np.uint8)
        union_mask = np.zeros((height, width), dtype=np.uint8)
        overlap_masks = []  # 存储有重叠的mask
        
        if len(annotations) == 0:
            pass
        else:
            for idx, ann in enumerate(annotations):
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                
                union_mask = np.maximum(union_mask, mask)
                
                # 检测是否有重叠（与已分配的区域重叠）
                has_overlap = (instance_mask_map > 0) & (mask > 0)
                
                if has_overlap.any():
                    # 有重叠：保存这个mask，后续单独处理
                    overlap_masks.append(mask)
                else:
                    # 无重叠：正常分配ID
                    instance_mask_map[mask > 0] = idx + 1
        
        # 非重叠区域的边缘（从实例掩码图提取）
        if instance_mask_map.max() > 0:
            edge_from_instance = cv2.morphologyEx(instance_mask_map, cv2.MORPH_GRADIENT, self.kernel)
            edge_from_instance = (edge_from_instance > 0).astype(np.uint8)
        else:
            edge_from_instance = np.zeros((height, width), dtype=np.uint8)
        
        # 重叠区域的边缘（单独提取）
        edge_from_overlap = np.zeros((height, width), dtype=np.uint8)
        for mask in overlap_masks:
            edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
            edge_from_overlap = np.logical_or(edge_from_overlap, edge).astype(np.uint8)
        
        # 合并两种边缘
        combined_edge_map = np.logical_or(edge_from_instance, edge_from_overlap).astype(np.uint8)
        
        # 生成多尺度边缘图
        edge_maps = {'original': combined_edge_map}
        for size in edge_sizes:
            edge_float = combined_edge_map.astype(np.float32)
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        return edge_maps, union_mask, instance_mask_map
    
    def process_single_image_speed_test(self, json_path, methods=['a', 'b', 'c']):
        """
        纯速度测试：只统计边缘提取时间（从JSON开始到返回边缘图）
        不加载图像，不生成可视化，不保存文件
        
        Args:
            methods: 要测试的方法列表，如 ['b'] 只测试Method B
        
        Returns:
            dict: 包含timings和edge_pixels统计
        """
        timings = {}
        edge_pixels = {}
        
        if 'a' in methods:
            # Method A: 从JSON开始，统计完整时间
            t0 = time.time()
            edge_maps_a, union_mask_a = self.method_a_baseline(json_path)
            timings['method_a'] = time.time() - t0
            edge_pixels['method_a'] = edge_maps_a[256].sum()
        
        if 'b' in methods:
            # Method B: 从JSON开始，统计完整时间
            t0 = time.time()
            edge_maps_b, union_mask_b = self.method_b_per_instance(json_path)
            timings['method_b'] = time.time() - t0
            edge_pixels['method_b'] = edge_maps_b[256].sum()
        
        if 'c' in methods:
            # Method C: 从JSON开始，统计完整时间
            t0 = time.time()
            edge_maps_c, union_mask_c, instance_mask_map_c = self.method_c_instance_mask(json_path)
            timings['method_c'] = time.time() - t0
            edge_pixels['method_c'] = edge_maps_c[256].sum()
            
            # Method C Optimized: 质量优化版本
            t0 = time.time()
            edge_maps_c_opt, union_mask_c_opt, instance_mask_map_c_opt = self.method_c_instance_mask_optimized(json_path)
            timings['method_c_optimized'] = time.time() - t0
            edge_pixels['method_c_optimized'] = edge_maps_c_opt[256].sum()
        
        return {
            'timings': timings,
            'edge_pixels': edge_pixels
        }
    
    def process_single_image_visualization(self, image_path, json_path, output_dir, show_method_b_only=False):
        """
        可视化：不计时，只生成对比图
        
        Args:
            show_method_b_only: 如果True，只显示原图+A+B+B优化（4列）
        """
        image_id = Path(image_path).stem
        
        try:
            # 加载图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if show_method_b_only:
                # 只运行A和B方法（B的原始版本和优化版本）
                edge_maps_a, union_mask_a = self.method_a_baseline(json_path)
                edge_maps_b_original, union_mask_b_orig = self.method_b_per_instance_original(json_path)  # 原始版本
                edge_maps_b_optimized, union_mask_b_opt = self.method_b_per_instance(json_path)  # 优化版本
                
                # 生成可视化（4列：原图+A+B原始+B优化）
                self.visualize_comparison_method_b_only(
                    image_rgb,
                    edge_maps_a[256],
                    edge_maps_b_original[256],  # 原始版本
                    edge_maps_b_optimized[256],  # 优化版本
                    output_dir / f"{image_id}_comparison.png"
                )
            else:
                # 运行所有方法（原始5列布局）
                edge_maps_a, union_mask_a = self.method_a_baseline(json_path)
                edge_maps_b, union_mask_b = self.method_b_per_instance(json_path)  # 已优化
                edge_maps_c, union_mask_c, instance_mask_map = self.method_c_instance_mask(json_path)
                edge_maps_c_opt, union_mask_c_opt, instance_mask_map_opt = self.method_c_instance_mask_optimized(json_path)
                
                # 生成可视化（5列：原图+A+B+C+优化C）
                self.visualize_comparison(
                    image_rgb,
                    edge_maps_a[256],
                    edge_maps_b[256],
                    edge_maps_c[256],
                    edge_maps_c_opt[256],
                    instance_mask_map,
                    output_dir / f"{image_id}_comparison.png"
                )
            
            return {'success': True, 'image_id': image_id}
            
        except Exception as e:
            return {'success': False, 'image_id': image_id, 'error': str(e)}
    
    def visualize_comparison(self, image, edge_a, edge_b, edge_c, edge_c_opt, instance_mask_map, output_path):
        """
        Visualize comparison: 5 subplots (Original, Method A, Method B, Method C, Method C Optimized)
        """
        fig = plt.figure(figsize=(20, 4))
        gs = GridSpec(1, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image)
        ax0.set_title("Original Image", fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        # Method A: Baseline
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(edge_a, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f"Method A (Baseline)\nUnion mask → Edge\nPixels: {edge_a.sum()}", 
                      fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Method B: Per instance
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(edge_b, cmap='hot', vmin=0, vmax=1)
        ax2.set_title(f"Method B (Per Instance)\nEdge per mask → Merge\nPixels: {edge_b.sum()}", 
                      fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Method C: Instance map
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.imshow(edge_c, cmap='hot', vmin=0, vmax=1)
        ax3.set_title(f"Method C (Instance Map)\nInstance map → Morphology\nEdge Pixels: {edge_c.sum()}", 
                      fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # Method C Optimized: 质量优化版本
        ax4 = fig.add_subplot(gs[0, 4])
        ax4.imshow(edge_c_opt, cmap='hot', vmin=0, vmax=1)
        ax4.set_title(f"Method C (Optimized)\nHandle overlap regions\nEdge Pixels: {edge_c_opt.sum()}", 
                      fontsize=11, fontweight='bold', color='green')
        ax4.axis('off')
        
        plt.suptitle(f'Edge Extraction Comparison - {Path(output_path).stem}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_comparison_method_b_only(self, image, edge_a, edge_b_original, edge_b_optimized, output_path):
        """
        Visualize comparison: 4 subplots (Original, Method A, Method B Original, Method B Optimized)
        用于对比B方法优化前后的效果
        """
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image)
        ax0.set_title("Original Image", fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        # Method A: Baseline
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(edge_a, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f"Method A (Baseline)\nUnion mask → Edge\nPixels: {edge_a.sum()}", 
                      fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Method B Original: 原始版本（logical_or + 每次转换）
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(edge_b_original, cmap='hot', vmin=0, vmax=1)
        ax2.set_title(f"Method B (Original)\nlogical_or + astype\nPixels: {edge_b_original.sum()}", 
                      fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Method B Optimized: 优化后的版本（bitwise_or + 二值化）
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.imshow(edge_b_optimized, cmap='hot', vmin=0, vmax=1)
        ax3.set_title(f"Method B (Optimized)\nbitwise_or + binary\nPixels: {edge_b_optimized.sum()}", 
                      fontsize=11, fontweight='bold', color='green')
        ax3.axis('off')
        
        plt.suptitle(f'Edge Extraction Comparison (Method B) - {Path(output_path).stem}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_speed_test(self, data_dir, max_images=100, methods=['a', 'b', 'c']):
        """
        纯速度测试：只统计边缘提取时间
        根据methods参数选择要测试的方法
        
        Args:
            methods: 要测试的方法列表，如 ['b'] 只测试Method B
        """
        data_path = Path(data_dir)
        
        # 获取JSON文件列表
        json_files = sorted([f for f in data_path.glob("*.json")])[:max_images]
        
        print(f"📁 Total JSON files: {len(json_files)}")
        print(f"⏱️  Running speed test (edge extraction only)...")
        print(f"📋 Testing methods: {', '.join([f'Method {m.upper()}' for m in methods])}")
        
        success_count = 0
        error_count = 0
        
        # 根据methods动态创建统计字典
        total_timings = {}
        edge_pixel_stats = {}
        
        if 'a' in methods:
            total_timings['method_a'] = 0.0
            edge_pixel_stats['method_a'] = []
        if 'b' in methods:
            total_timings['method_b'] = 0.0
            edge_pixel_stats['method_b'] = []
        if 'c' in methods:
            total_timings['method_c'] = 0.0
            total_timings['method_c_optimized'] = 0.0
            edge_pixel_stats['method_c'] = []
            edge_pixel_stats['method_c_optimized'] = []
        
        for json_file in tqdm(json_files, desc="Speed test"):
            try:
                result = self.process_single_image_speed_test(json_file, methods=methods)
                
                success_count += 1
                
                # 累加时间（只统计边缘提取时间）
                for method in result['timings']:
                    if method in total_timings:
                        total_timings[method] += result['timings'][method]
                
                # 累加边缘像素统计
                for method in result['edge_pixels']:
                    if method in edge_pixel_stats:
                        edge_pixel_stats[method].append(result['edge_pixels'][method])
                
                # 每10张打印一次进度
                if success_count % 10 == 0:
                    timing_str = ", ".join([f"{k}: {result['timings'][k]:.4f}s" 
                                           for k in result['timings']])
                    print(f"\n✅ {Path(json_file).stem}: {timing_str}")
            except Exception as e:
                error_count += 1
                print(f"❌ {Path(json_file).stem}: {e}")
        
        # 打印统计
        print(f"\n🎉 Speed Test Complete!")
        print(f"✅ Success: {success_count} images")
        print(f"❌ Failed: {error_count} images")
        
        print(f"\n⏱️  Edge Extraction Timing (per image):")
        for method, total_time in total_timings.items():
            avg_time = total_time / success_count if success_count > 0 else 0
            print(f"  {method}: {avg_time:.4f}s (total: {total_time:.2f}s)")
        
        print(f"\n📊 Edge Pixel Statistics (256×256):")
        for method, pixel_list in edge_pixel_stats.items():
            if pixel_list:
                avg_pixels = np.mean(pixel_list)
                std_pixels = np.std(pixel_list)
                print(f"  {method}: mean={avg_pixels:.1f}, std={std_pixels:.1f}, "
                      f"min={np.min(pixel_list):.0f}, max={np.max(pixel_list):.0f}")
        
        # 估算100个epoch的时间（假设每张图都要处理）
        if success_count > 0:
            images_per_epoch = success_count
            print(f"\n📈 Estimated Time for 100 Epochs:")
            for method, total_time in total_timings.items():
                avg_time = total_time / success_count
                time_per_epoch = avg_time * images_per_epoch
                time_100_epochs = time_per_epoch * 100
                print(f"  {method}: {avg_time:.4f}s/img → "
                      f"{time_per_epoch:.2f}s/epoch ({time_per_epoch/60:.2f} min) → "
                      f"{time_100_epochs/3600:.2f} hours for 100 epochs")
        
        # 速度对比分析
        print(f"\n📊 Speed Comparison:")
        if success_count > 0:
            if 'method_a' in total_timings and 'method_b' in total_timings:
                avg_a = total_timings['method_a'] / success_count
                avg_b = total_timings['method_b'] / success_count
                print(f"  Method A (Baseline):      {avg_a:.4f}s (baseline)")
                print(f"  Method B (Per Instance):  {avg_b:.4f}s ({avg_b/avg_a:.2f}x slower)")
            
            if 'method_b' in total_timings and 'method_a' not in total_timings:
                # 只测试B方法时
                avg_b = total_timings['method_b'] / success_count
                print(f"  Method B (Optimized):      {avg_b:.4f}s per image")
            
            if 'method_c' in total_timings:
                avg_c = total_timings['method_c'] / success_count
                avg_c_opt = total_timings['method_c_optimized'] / success_count
                if 'method_a' in total_timings:
                    avg_a = total_timings['method_a'] / success_count
                    print(f"  Method C (Instance Map):  {avg_c:.4f}s ({avg_c/avg_a:.2f}x slower)")
                    print(f"  Method C (Optimized):      {avg_c_opt:.4f}s ({avg_c_opt/avg_a:.2f}x slower)")
                else:
                    print(f"  Method C (Instance Map):  {avg_c:.4f}s")
                    print(f"  Method C (Optimized):      {avg_c_opt:.4f}s ({avg_c_opt/avg_c:.2f}x slower)")
    
    def batch_visualization(self, data_dir, output_dir, max_images=20, update_existing=False, show_method_b_only=False):
        """
        可视化：前N张图生成对比图
        不计时，只用于质量对比
        如果update_existing=True，会更新已存在的可视化图（拼接优化C结果）
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件列表
        image_files = sorted(list(data_path.glob("*.jpg")))[:max_images]
        
        print(f"\n📁 Generating visualizations for {len(image_files)} images...")
        print(f"📁 Output directory: {output_path}")
        print(f"📁 Update existing: {update_existing}")
        
        success_count = 0
        error_count = 0
        
        for image_file in tqdm(image_files, desc="Visualization"):
            json_file = data_path / f"{image_file.stem}.json"
            
            if not json_file.exists():
                print(f"⚠️  Skip {image_file.stem}: JSON file not found")
                error_count += 1
                continue
            
            # 如果update_existing且文件已存在，先加载原图再拼接
            output_file = output_path / f"{image_file.stem}_comparison.png"
            if update_existing and output_file.exists():
                # 加载已存在的可视化图
                existing_img = plt.imread(str(output_file))
                
                # 运行优化后的Method C
                edge_maps_c_opt, _, _ = self.method_c_instance_mask_optimized(json_path=json_file)
                
                # 拼接：在最右边添加优化C的结果
                self.append_optimized_c_to_visualization(
                    output_file,
                    edge_maps_c_opt[256]
                )
                success_count += 1
            else:
                # 生成新的可视化图
                result = self.process_single_image_visualization(
                    image_file, json_file, output_path,
                    show_method_b_only=show_method_b_only
                )
                
                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"❌ {result['image_id']}: {result['error']}")
        
        print(f"\n🎉 Visualization Complete!")
        print(f"✅ Success: {success_count} images")
        print(f"❌ Failed: {error_count} images")
        print(f"📁 Output directory: {output_path}")
    
    def append_optimized_c_to_visualization(self, existing_img_path, edge_c_opt):
        """
        在已存在的可视化图的最右边拼接优化后的Method C结果
        如果原图是4列，扩展为5列；如果已经是5列，替换最后一列
        """
        # 加载已存在的图
        existing_img = plt.imread(str(existing_img_path))
        existing_height, existing_width = existing_img.shape[:2]
        
        # 如果是RGBA，转为RGB
        if existing_img.shape[2] == 4:
            existing_img = existing_img[:, :, :3]
        
        # 判断是4列还是5列（假设每列宽度相等）
        # 先尝试4列
        col_width_4 = existing_width // 4
        col_width_5 = existing_width // 5
        
        # 检查是否是4列（余数较小）
        remainder_4 = existing_width % 4
        remainder_5 = existing_width % 5
        
        if remainder_4 < remainder_5:
            # 原图是4列，需要扩展为5列
            is_4cols = True
            col_width = col_width_4
            new_width = col_width * 5
            new_img = np.zeros((existing_height, new_width, 3))
            # 复制原有的4列
            new_img[:, :existing_width] = existing_img
            # 在第5列位置添加优化C
            insert_pos = existing_width
        else:
            # 原图已经是5列，替换最后一列
            is_4cols = False
            col_width = col_width_5
            new_width = existing_width
            new_img = existing_img.copy()
            # 替换第5列
            insert_pos = col_width * 4
        
        # 准备优化C的边缘图（resize到列宽x高度）
        # edge_c_opt是256x256的
        target_width = insert_pos + col_width - insert_pos  # 实际需要插入的宽度
        edge_c_opt_resized = cv2.resize(
            edge_c_opt.astype(np.float32),
            (target_width, existing_height),
            interpolation=cv2.INTER_AREA
        )
        
        # 转换为RGB（使用hot colormap）
        # 归一化到0-1
        edge_normalized = edge_c_opt_resized.astype(np.float32)
        if edge_normalized.max() > 1:
            edge_normalized = edge_normalized / 255.0
        
        # 使用hot colormap转换为RGB
        edge_c_opt_rgb = plt.cm.hot(edge_normalized)[:, :, :3]
        # 确保值在[0, 1]范围
        edge_c_opt_rgb = np.clip(edge_c_opt_rgb, 0, 1)
        
        # 确保尺寸匹配（可能因为除法导致的小误差）
        actual_width = new_img.shape[1] - insert_pos
        if edge_c_opt_rgb.shape[1] != actual_width:
            edge_c_opt_rgb = cv2.resize(
                edge_c_opt_rgb.astype(np.float32),
                (actual_width, existing_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 在指定位置插入/替换优化C的结果
        new_img[:, insert_pos:insert_pos+edge_c_opt_rgb.shape[1]] = edge_c_opt_rgb
        
        # 保存（覆盖原文件）
        plt.imsave(existing_img_path, new_img, dpi=150)
    
    def batch_test_method_c_only(self, data_dir, max_images=100):
        """
        只测试Method C（原始和优化版本）的速度和质量
        """
        data_path = Path(data_dir)
        json_files = sorted([f for f in data_path.glob("*.json")])[:max_images]
        
        print(f"📁 Testing Method C only on {len(json_files)} images...")
        
        success_count = 0
        error_count = 0
        
        total_timings = {
            'method_c': 0.0,
            'method_c_optimized': 0.0,
        }
        
        edge_pixel_stats = {
            'method_c': [],
            'method_c_optimized': [],
        }
        
        for json_file in tqdm(json_files, desc="Method C test"):
            try:
                # Method C原始版本
                t0 = time.time()
                edge_maps_c, _, _ = self.method_c_instance_mask(json_file)
                time_c = time.time() - t0
                
                # Method C优化版本
                t0 = time.time()
                edge_maps_c_opt, _, _ = self.method_c_instance_mask_optimized(json_file)
                time_c_opt = time.time() - t0
                
                success_count += 1
                total_timings['method_c'] += time_c
                total_timings['method_c_optimized'] += time_c_opt
                edge_pixel_stats['method_c'].append(edge_maps_c[256].sum())
                edge_pixel_stats['method_c_optimized'].append(edge_maps_c_opt[256].sum())
                
                if success_count % 10 == 0:
                    print(f"\n✅ {Path(json_file).stem}: "
                          f"Method C: {time_c:.4f}s, "
                          f"Method C Opt: {time_c_opt:.4f}s")
            except Exception as e:
                error_count += 1
                print(f"❌ {Path(json_file).stem}: {e}")
        
        # 打印统计
        print(f"\n🎉 Method C Test Complete!")
        print(f"✅ Success: {success_count} images")
        print(f"❌ Failed: {error_count} images")
        
        print(f"\n⏱️  Timing (per image):")
        for method, total_time in total_timings.items():
            avg_time = total_time / success_count if success_count > 0 else 0
            print(f"  {method}: {avg_time:.4f}s (total: {total_time:.2f}s)")
        
        print(f"\n📊 Edge Pixel Statistics (256×256):")
        for method, pixel_list in edge_pixel_stats.items():
            if pixel_list:
                avg_pixels = np.mean(pixel_list)
                std_pixels = np.std(pixel_list)
                print(f"  {method}: mean={avg_pixels:.1f}, std={std_pixels:.1f}, "
                      f"min={np.min(pixel_list):.0f}, max={np.max(pixel_list):.0f}")
        
        if success_count > 0:
            avg_c = total_timings['method_c'] / success_count
            avg_c_opt = total_timings['method_c_optimized'] / success_count
            print(f"\n📊 Speed Comparison:")
            print(f"  Method C:           {avg_c:.4f}s")
            print(f"  Method C Optimized: {avg_c_opt:.4f}s ({avg_c_opt/avg_c:.2f}x slower)")
            
            avg_pixels_c = np.mean(edge_pixel_stats['method_c'])
            avg_pixels_c_opt = np.mean(edge_pixel_stats['method_c_optimized'])
            print(f"\n📊 Quality Comparison:")
            print(f"  Method C pixels:           {avg_pixels_c:.1f}")
            print(f"  Method C Optimized pixels: {avg_pixels_c_opt:.1f} ({avg_pixels_c_opt/avg_pixels_c:.2f}x more)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Edge extraction method comparison experiment")
    parser.add_argument("--data-dir", type=str, required=True, help="SA-1B data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of images for speed test")
    parser.add_argument("--kernel-size", type=int, default=3, help="Edge extraction kernel size")
    parser.add_argument("--device", type=str, default="cuda:5", help="GPU device (e.g., cuda:0, cuda:5)")
    parser.add_argument("--visualization-only", action='store_true', help="Only generate visualizations, skip speed test")
    parser.add_argument("--speed-test-only", action='store_true', help="Only run speed test, skip visualizations")
    parser.add_argument("--update-visualizations", action='store_true', help="Update existing visualizations by appending optimized Method C")
    parser.add_argument("--test-method-c-only", action='store_true', help="Only test Method C (original and optimized) on 100 images")
    parser.add_argument("--test-method-b-only", action='store_true', help="Only test Method B (optimized) on 100 images")
    parser.add_argument("--show-method-b-only", action='store_true', help="Visualization: show only Method A and B comparison (4 columns)")
    parser.add_argument("--test-bbox", action='store_true', help="Test bbox extraction from SA-1B JSON files")
    parser.add_argument("--num-bbox-test", type=int, default=50, help="Number of JSON files for bbox extraction speed test")
    parser.add_argument("--num-bbox-vis", type=int, default=5, help="Number of files to visualize bboxes")
    parser.add_argument("--bbox-start-index", type=int, default=1000, help="Start index for bbox visualization (skip first N files)")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device ID for bbox extraction (超参数)")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="NMS IoU threshold for removing overlapping boxes (建议0.3-0.7)")
    
    args = parser.parse_args()
    
    # Set CUDA device FIRST (before any other operations)
    print("=== Setting CUDA Device ===")
    device = set_cuda_device(args.device)
    
    config = {
        'kernel_size': args.kernel_size,
        'device': str(device)
    }
    
    print("\n=== Edge Extraction Method Comparison ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Max images (speed test): {args.max_images}")
    
    # 根据参数选择执行模式
    if args.test_bbox:
        # Bbox 提取测试和可视化
        print("\n" + "="*60)
        print("Bbox Extraction Test")
        print("="*60)
        
        bbox_extractor = SA1BInstanceBoxExtractor(
            min_area_threshold=1000,
            min_iou_threshold=0.90,
            nms_iou_threshold=args.nms_iou_threshold,
            use_cuda=True,
            device=device,
            cuda_device_id=args.cuda_device_id,
        )
        
        # 速度测试
        print(f"\n阶段1: 速度测试 ({args.num_bbox_test} 个文件)")
        print("-"*60)
        bbox_extractor.batch_extract_test(args.data_dir, num_files=args.num_bbox_test)
        
        # 可视化
        print(f"\n阶段2: 可视化 ({args.num_bbox_vis} 个文件)")
        print("-"*60)
        vis_output_dir = Path(args.output_dir) / "bbox_visualizations"
        bbox_extractor.batch_visualize(
            args.data_dir,
            vis_output_dir,
            num_visualize=args.num_bbox_vis,
            start_index=args.bbox_start_index
        )
        print(f"\n✅ 可视化结果保存到: {vis_output_dir}")
        
    elif args.test_method_c_only:
        # 只测试Method C（原始和优化版本）
        extractor = EdgeExtractionComparison(config)
        print("\n" + "="*60)
        print("Testing Method C Only (100 images)")
        print("="*60)
        extractor.batch_test_method_c_only(args.data_dir, args.max_images)
        if not args.speed_test_only:
            print("\n" + "="*60)
            print("Phase 2: Visualization")
            print("="*60)
            vis_output_dir = Path(args.output_dir) / "visualizations"
            extractor.batch_visualization(
                args.data_dir, 
                vis_output_dir, 
                max_images=20,
                update_existing=args.update_visualizations,
                show_method_b_only=args.show_method_b_only
            )
    elif args.test_method_b_only:
        # 只测试Method B（优化版本）
        extractor = EdgeExtractionComparison(config)
        print("\n" + "="*60)
        print("Testing Method B Only (100 images)")
        print("="*60)
        extractor.batch_speed_test(args.data_dir, args.max_images, methods=['b'])
    else:
        # 创建边缘提取对比器
        extractor = EdgeExtractionComparison(config)
        
        if not args.visualization_only:
            print("\n" + "="*60)
            print("Phase 1: Speed Test")
            print("="*60)
            extractor.batch_speed_test(args.data_dir, args.max_images)
        
        if not args.speed_test_only:
            print("\n" + "="*60)
            print("Phase 2: Visualization")
            print("="*60)
            vis_output_dir = Path(args.output_dir) / "visualizations"
            extractor.batch_visualization(
                args.data_dir, 
                vis_output_dir, 
                max_images=20,
                update_existing=args.update_visualizations,
            show_method_b_only=args.show_method_b_only
        )


if __name__ == "__main__":
    main()

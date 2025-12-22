#!/usr/bin/env python3
"""
SA-1B 实例框提取器
从 SA-1B JSON 标注文件中快速提取实例框（2个大框+1个中框）
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from tqdm import tqdm
import torch
import torchvision.ops


class SA1BInstanceBoxExtractor:
    """SA-1B 实例框提取器：提取2个大框+1个中框"""
    
    def __init__(
        self,
        min_area_threshold: int = 1000,
        min_iou_threshold: float = 0.90,
        nms_iou_threshold: float = 0.5,
        use_cuda: bool = True,
        device: Optional[torch.device] = None,
        cuda_device_id: int = 0,
        use_composite_score: bool = True,
        area_weight: float = 0.2,  # 降低面积权重，优先质量和稳定性
        iou_weight: float = 0.5,   # 提高IoU权重，优先高质量掩码
        stability_weight: float = 0.3,  # 提高稳定性权重，过滤不稳定掩码
        two_stage_nms: bool = True,
        mask_size: int = 256,
        # 新增：完整性权重与动态K选择
        use_integrity: bool = True,
        score_threshold: float = 0.80,
        max_instances: int = 5,
        # 新增：背景/天空抑制规则
        max_area_ratio: float = 0.50,  # 掩码面积占整图比例的上限（大于则视为背景/天空）
        reject_top_band: bool = True,  # 过滤顶部横向大带状区域
        top_edge_margin_frac: float = 0.02,  # 距离顶部阈值（相对高度）
        min_top_band_width_frac: float = 0.60,  # 顶部带状区域需覆盖的最小宽度比例
        min_top_band_area_ratio: float = 0.20,  # 顶部带状区域的最小面积占比
    ):
        """
        Args:
            min_area_threshold: 最小掩码面积（像素），过滤小物体
            min_iou_threshold: 最小 predicted_iou，过滤低质量掩码
            nms_iou_threshold: NMS 的 IoU 阈值，合并重叠框（重要！建议0.3-0.7）
            use_cuda: 是否使用CUDA加速（批量处理时有效）
            device: CUDA设备，如果为None则自动检测
            cuda_device_id: CUDA设备ID（超参数，默认0）
            use_composite_score: 是否使用综合评分（area+iou+stability）
            area_weight: 面积权重（综合评分）
            iou_weight: predicted_iou权重（综合评分）
            stability_weight: stability_score权重（综合评分）
            two_stage_nms: 是否使用两阶段NMS
            mask_size: 掩码保存的尺寸（默认256x256）
            use_integrity: 是否使用“实例完整性”权重（单点/全图视为整体）
            score_threshold: 动态K的分数阈值（>=该分数的实例被保留）
            max_instances: 最大实例数（默认5）
        """
        self.min_area = min_area_threshold
        self.min_iou = min_iou_threshold
        self.nms_iou = nms_iou_threshold
        self.use_composite_score = use_composite_score
        self.area_weight = area_weight
        self.iou_weight = iou_weight
        self.stability_weight = stability_weight
        self.two_stage_nms = two_stage_nms
        self.mask_size = mask_size
        self.use_integrity = use_integrity
        self.score_threshold = score_threshold
        self.max_instances = max_instances
        # 背景抑制参数
        self.max_area_ratio = max_area_ratio
        self.reject_top_band = reject_top_band
        self.top_edge_margin_frac = top_edge_margin_frac
        self.min_top_band_width_frac = min_top_band_width_frac
        self.min_top_band_area_ratio = min_top_band_area_ratio
        
        # CUDA配置（支持超参数设置）
        if use_cuda and torch.cuda.is_available():
            if device is None:
                # 使用超参数指定的设备ID
                if cuda_device_id >= torch.cuda.device_count():
                    print(f"⚠️  设备 cuda:{cuda_device_id} 不存在，使用 cuda:0")
                    cuda_device_id = 0
                self.device = torch.device(f'cuda:{cuda_device_id}')
            else:
                self.device = device
            self.use_cuda = True
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False
        
        print(f"✅ SA1BInstanceBoxExtractor 初始化")
        print(f"  CUDA加速: {self.use_cuda}")
        if self.use_cuda:
            print(f"  设备: {self.device}")
        print(f"  NMS阈值: {self.nms_iou}")
        print(f"  综合评分: {self.use_composite_score} (Area:{self.area_weight}, IoU:{self.iou_weight}, Stability:{self.stability_weight})")
        print(f"  两阶段NMS: {self.two_stage_nms}")
        print(f"  掩码尺寸: {self.mask_size}x{self.mask_size}")
        print(f"  完整性权重: {self.use_integrity}")
        print(f"  动态K: 阈值={self.score_threshold}, 最大实例={self.max_instances}")
        print(f"  背景抑制: max_area_ratio={self.max_area_ratio}, reject_top_band={self.reject_top_band}")
        print(f"\n📌 选框逻辑说明:")
        print(f"  1. 过滤: 面积≥{self.min_area}, IoU≥{self.min_iou}")
        print(f"  2. 评分: 综合评分排序 (面积+质量+稳定性){' × 完整性' if self.use_integrity else ''}")
        print(f"  3. NMS: {'两阶段' if self.two_stage_nms else '单阶段'} 去重叠")
        print(f"  4. 选择: 动态K(分数≥{self.score_threshold})，最多{self.max_instances} 个实例")
    
    def extract_top_boxes_simple(
        self, 
        json_path: str
    ) -> Dict[str, Any]:
        """
        提取2个大框 + 1个中框（第3大的）
        使用综合评分和两阶段NMS优化
        自动处理不足3个的情况
        
        Returns:
            {
                'large_boxes': [[x, y, w, h], ...],  # 最多2个
                'medium_boxes': [[x, y, w, h]],      # 最多1个
                'masks': np.array,  # [3, 256, 256] 掩码，每个通道对应一个框
                'annotation_indices': [int, int, int],  # 对应原始annotation的索引
                'total_available': int
            }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        image_height = data['image']['height']
        image_width = data['image']['width']
        
        # 过滤并提取框（带面积信息）
        boxes_list = []
        areas_list = []
        ious_list = []
        stability_list = []
        rle_list = []  # 保存RLE用于后续生成掩码
        integrity_list = []  # 新增：实例完整性
        
        # 获取图像尺寸（用于面积占比与带状检测）
        image_info = data.get('image', {})
        image_height = int(image_info.get('height', 0) or image_info.get('h', 0) or 0)
        image_width = int(image_info.get('width', 0) or image_info.get('w', 0) or 0)
        image_area = float(max(1, image_height * image_width))

        for ann in annotations:
            # 直接从 RLE 提取框（非常快！）
            bbox = mask_utils.toBbox(ann['segmentation'])
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            area_val = float(ann['area'])

            # 背景/天空抑制：面积占比过大或顶部横向大带状
            area_ratio = area_val / image_area
            is_reject = False
            if area_ratio >= self.max_area_ratio:
                is_reject = True
            if (not is_reject) and self.reject_top_band and image_height > 0 and image_width > 0:
                top_touch = (y <= self.top_edge_margin_frac * image_height)
                width_frac = w / image_width
                if top_touch and width_frac >= self.min_top_band_width_frac and area_ratio >= self.min_top_band_area_ratio:
                    is_reject = True
            if is_reject:
                # 跳过疑似天空/大背景区域
                continue

            boxes_list.append([x, y, w, h])
            areas_list.append(area_val)
            ious_list.append(ann.get('predicted_iou', 0.0))
            stability_list.append(ann.get('stability_score', 0.0))
            rle_list.append(ann['segmentation'])  # 保存RLE
            # 完整性：单点(point_coords长度为1或2) 或 未裁剪(crop_n_layers=0) 视为整体
            point_coords = ann.get('point_coords', [])
            crop_n_layers = ann.get('crop_n_layers', None)
            is_high_integrity = False
            if isinstance(point_coords, list) and 0 < len(point_coords) < 3:
                is_high_integrity = True
            if crop_n_layers is not None and crop_n_layers == 0:
                is_high_integrity = True
            integrity_list.append(1.0 if is_high_integrity else 0.5)
        
        if len(boxes_list) == 0:
            return {
                'large_boxes': [],
                'medium_boxes': [],
                'masks': np.zeros((self.max_instances, self.mask_size, self.mask_size), dtype=np.uint8),
                'annotation_indices': [],
                'total_available': 0
            }
        
        # 使用CUDA加速过滤和排序（批量处理时更高效）
        if self.use_cuda and len(boxes_list) > 100:
            # 批量处理：转为tensor进行向量化操作
            areas_tensor = torch.tensor(areas_list, dtype=torch.float32, device=self.device)
            ious_tensor = torch.tensor(ious_list, dtype=torch.float32, device=self.device)
            stability_tensor = torch.tensor(stability_list, dtype=torch.float32, device=self.device)
            integrity_tensor = torch.tensor(integrity_list, dtype=torch.float32, device=self.device)
            
            # 向量化过滤
            area_mask = areas_tensor >= self.min_area
            iou_mask = ious_tensor >= self.min_iou
            valid_mask = area_mask & iou_mask
            
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return {
                    'large_boxes': [],
                    'medium_boxes': [],
                    'masks': np.zeros((self.max_instances, self.mask_size, self.mask_size), dtype=np.uint8),
                    'annotation_indices': [],
                    'total_available': 0
                }
            
            # 获取有效的数据
            valid_areas = areas_tensor[valid_indices]
            valid_ious = ious_tensor[valid_indices]
            valid_stability = stability_tensor[valid_indices]
            valid_integrity = integrity_tensor[valid_indices]
            valid_indices_cpu = valid_indices.cpu().numpy()
            
            # 综合评分策略 + 完整性乘子
            if self.use_composite_score:
                # 归一化各项指标（避免量纲差异）
                areas_norm = (valid_areas - valid_areas.min()) / (valid_areas.max() - valid_areas.min() + 1e-8)
                ious_norm = valid_ious  # 已在[0,1]
                stability_norm = valid_stability  # 已在[0,1]
                composite_scores = (
                    self.area_weight * areas_norm +
                    self.iou_weight * ious_norm +
                    self.stability_weight * stability_norm
                )
                if self.use_integrity:
                    composite_scores = composite_scores * valid_integrity
                scores_tensor = composite_scores
            else:
                scores_tensor = valid_areas * (valid_integrity if self.use_integrity else 1.0)
            
            # 转换为 [x1, y1, x2, y2] 格式用于NMS
            boxes_xywh = np.array([boxes_list[i] for i in valid_indices_cpu], dtype=np.float32)
            boxes_xyxy = boxes_xywh.copy()
            boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]  # x2 = x + w
            boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]  # y2 = y + h
            
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
            
            # 两阶段NMS
            if self.two_stage_nms:
                keep_indices_stage1 = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.7)
                boxes_tensor_stage1 = boxes_tensor[keep_indices_stage1]
                scores_tensor_stage1 = scores_tensor[keep_indices_stage1]
                keep_indices_stage2 = torchvision.ops.nms(boxes_tensor_stage1, scores_tensor_stage1, self.nms_iou)
                keep_indices = keep_indices_stage1[keep_indices_stage2]
            else:
                keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, self.nms_iou)
            
            kept_boxes_xywh = boxes_xywh[keep_indices.cpu().numpy()]
            kept_scores = scores_tensor[keep_indices].cpu().numpy()
            kept_original_indices = valid_indices_cpu[keep_indices.cpu().numpy()]
            
            # 按分数排序
            order = np.argsort(kept_scores)[::-1]
            valid_boxes = []
            for i in order:
                valid_boxes.append({
                    'box': kept_boxes_xywh[i].tolist(),
                    'score': float(kept_scores[i]),
                    'original_idx': int(kept_original_indices[i])
                })
        else:
            # CPU版本：
            valid_boxes_data = []
            for i in range(len(boxes_list)):
                if areas_list[i] < self.min_area:
                    continue
                if ious_list[i] < self.min_iou:
                    continue
                boxes_xywh = boxes_list[i]
                boxes_xyxy = [boxes_xywh[0], boxes_xywh[1], boxes_xywh[0] + boxes_xywh[2], boxes_xywh[1] + boxes_xywh[3]]
                valid_boxes_data.append({
                    'box': boxes_xywh,
                    'box_xyxy': boxes_xyxy,
                    'area': areas_list[i],
                    'iou': ious_list[i],
                    'stability': stability_list[i],
                    'integrity': integrity_list[i],
                    'original_idx': i
                })
            if len(valid_boxes_data) == 0:
                return {
                    'large_boxes': [],
                    'medium_boxes': [],
                    'masks': np.zeros((self.max_instances, self.mask_size, self.mask_size), dtype=np.uint8),
                    'annotation_indices': [],
                    'total_available': 0
                }
            if self.use_composite_score:
                areas_arr = np.array([x['area'] for x in valid_boxes_data])
                ious_arr = np.array([x['iou'] for x in valid_boxes_data])
                stability_arr = np.array([x['stability'] for x in valid_boxes_data])
                integrity_arr = np.array([x['integrity'] for x in valid_boxes_data])
                areas_norm = (areas_arr - areas_arr.min()) / (areas_arr.max() - areas_arr.min() + 1e-8)
                composite_scores = (
                    self.area_weight * areas_norm +
                    self.iou_weight * ious_arr +
                    self.stability_weight * stability_arr
                )
                if self.use_integrity:
                    composite_scores = composite_scores * integrity_arr
            else:
                composite_scores = np.array([x['area'] for x in valid_boxes_data]) * (
                    np.array([x['integrity'] for x in valid_boxes_data]) if self.use_integrity else 1.0
                )
            for i, s in enumerate(composite_scores):
                valid_boxes_data[i]['score'] = float(s)
            # 排序
            valid_boxes_data.sort(key=lambda x: x['score'], reverse=True)
            # NMS（两阶段）
            boxes_xyxy = torch.tensor([x['box_xyxy'] for x in valid_boxes_data], dtype=torch.float32)
            scores = torch.tensor([x['score'] for x in valid_boxes_data], dtype=torch.float32)
            if self.two_stage_nms:
                keep1 = torchvision.ops.nms(boxes_xyxy, scores, 0.7)
                boxes_t1 = boxes_xyxy[keep1]
                scores_t1 = scores[keep1]
                keep2 = torchvision.ops.nms(boxes_t1, scores_t1, self.nms_iou)
                keep_indices = keep1[keep2].numpy()
            else:
                keep_indices = torchvision.ops.nms(boxes_xyxy, scores, self.nms_iou).numpy()
            valid_boxes = [{
                'box': valid_boxes_data[i]['box'],
                'score': valid_boxes_data[i]['score'],
                'original_idx': valid_boxes_data[i]['original_idx']
            } for i in keep_indices]
            valid_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        total = len(valid_boxes)
        
        # 动态K：按阈值保留，最多max_instances
        selected_boxes = []
        selected_indices = []
        for item in valid_boxes:
            if item['score'] >= self.score_threshold and len(selected_boxes) < self.max_instances:
                selected_boxes.append(item['box'])
                selected_indices.append(item['original_idx'])
            else:
                if len(selected_boxes) >= self.max_instances:
                    break
                # 当遇到第一个低于阈值的，后续更低，直接停止
                if item['score'] < self.score_threshold:
                    break
        if len(selected_boxes) == 0 and total > 0:
            # 至少保留一个
            selected_boxes.append(valid_boxes[0]['box'])
            selected_indices.append(valid_boxes[0]['original_idx'])
        
        masks = self._generate_masks(rle_list, selected_indices)
        
        return {
            'boxes': selected_boxes,
            'masks': masks,
            'annotation_indices': selected_indices,
            'total_available': len(selected_boxes),
            # 兼容旧字段：取前3个做旧版切片（最多2大+1中）
            'large_boxes': selected_boxes[:2],
            'medium_boxes': selected_boxes[2:3],
        }
    
    def _generate_masks(
        self,
        rle_list: list,
        selected_indices: list,
        ) -> np.ndarray:
        """
        生成掩码：max_instances通道，每个通道对应一个框的掩码，resize到256x256
        """
        masks = np.zeros((self.max_instances, self.mask_size, self.mask_size), dtype=np.uint8)
        for channel_idx, ann_idx in enumerate(selected_indices[: self.max_instances]):
            if ann_idx >= len(rle_list):
                continue
            try:
                rle = rle_list[ann_idx]
                mask = mask_utils.decode(rle)
                mask_resized = cv2.resize(mask.astype(np.float32), (self.mask_size, self.mask_size), interpolation=cv2.INTER_AREA)
                masks[channel_idx] = (mask_resized > 0.5).astype(np.uint8)
            except Exception:
                continue
        return masks
    
    def batch_extract_test(
        self,
        json_dir: str,
        num_files: int = 50
    ) -> Dict[str, Any]:
        """
        批量测试提取速度
        
        Returns:
            包含速度和统计信息的字典
        """
        json_dir = Path(json_dir)
        json_files = sorted([f for f in json_dir.glob("sa_*.json")])[:num_files]
        
        print(f"📁 测试 {len(json_files)} 个 JSON 文件...")
        print(f"   使用CUDA: {self.use_cuda}, 设备: {self.device}")
        
        total_time = 0.0
        success_count = 0
        error_count = 0
        
        box_stats = {
            'total_files': len(json_files),
            'files_with_0_boxes': 0,
            'files_with_1_boxes': 0,
            'files_with_2_boxes': 0,
            'files_with_3_boxes': 0,
        }
        
        # 预热（避免首次运行慢）
        if len(json_files) > 0 and self.use_cuda:
            try:
                _ = self.extract_top_boxes_simple(str(json_files[0]))
            except:
                pass
        
        for json_file in tqdm(json_files, desc="提取bbox"):
            try:
                t0 = time.time()
                result = self.extract_top_boxes_simple(str(json_file))
                elapsed = time.time() - t0
                
                total_time += elapsed
                success_count += 1
                
                total_boxes = len(result['large_boxes']) + len(result['medium_boxes'])
                if total_boxes == 0:
                    box_stats['files_with_0_boxes'] += 1
                elif total_boxes == 1:
                    box_stats['files_with_1_boxes'] += 1
                elif total_boxes == 2:
                    box_stats['files_with_2_boxes'] += 1
                else:
                    box_stats['files_with_3_boxes'] += 1
                    
            except Exception as e:
                error_count += 1
                print(f"❌ 处理 {json_file.name} 时出错: {e}")
        
        avg_time = total_time / success_count if success_count > 0 else 0
        
        print(f"\n✅ 测试完成!")
        print(f"  成功: {success_count} 个文件")
        print(f"  失败: {error_count} 个文件")
        print(f"  平均速度: {avg_time*1000:.2f} ms/文件")
        print(f"  总耗时: {total_time:.2f} 秒")
        if self.use_cuda:
            print(f"  ⚡ CUDA加速: 启用")
        else:
            print(f"  ⚡ CUDA加速: 未启用")
        print(f"\n📊 框数量统计:")
        print(f"  0个框: {box_stats['files_with_0_boxes']} 个文件")
        print(f"  1个框: {box_stats['files_with_1_boxes']} 个文件")
        print(f"  2个框: {box_stats['files_with_2_boxes']} 个文件")
        print(f"  3个框: {box_stats['files_with_3_boxes']} 个文件")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'total_time': total_time,
            'success_count': success_count,
            'box_stats': box_stats,
            'use_cuda': self.use_cuda
        }
    
    def visualize_boxes_on_image(
        self,
        image_path: str,
        json_path: str,
        output_path: str
    ):
        """
        可视化：4个子图
        1. 原图+框标注（大框1红色，大框2绿色，中框蓝色）
        2. 掩码通道1（对应大框1）
        3. 掩码通道2（对应大框2）
        4. 掩码通道3（对应中框）
        """
        # 提取框和掩码
        result = self.extract_top_boxes_simple(json_path)
        large_boxes = result['large_boxes']
        medium_boxes = result['medium_boxes']
        masks = result['masks']  # [max_instances, 256, 256]
        
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建4个子图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 子图1：原图+框标注
        ax0 = axes[0]
        ax0.imshow(image_rgb)
        
        # 绘制大框1（红色，粗线）
        if len(large_boxes) > 0:
            x, y, w, h = large_boxes[0]
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=3, edgecolor='red', facecolor='none',
                label='Large Box 1'
            )
            ax0.add_patch(rect)
        
        # 绘制大框2（绿色，粗线）
        if len(large_boxes) > 1:
            x, y, w, h = large_boxes[1]
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=3, edgecolor='green', facecolor='none',
                label='Large Box 2'
            )
            ax0.add_patch(rect)
        
        # 绘制中框（蓝色，细线）
        if len(medium_boxes) > 0:
            x, y, w, h = medium_boxes[0]
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='blue', facecolor='none',
                label='Medium Box'
            )
            ax0.add_patch(rect)
        
        ax0.set_title(
            f"Original Image + Boxes\nLarge: {len(large_boxes)}, Medium: {len(medium_boxes)}",
            fontsize=11, fontweight='bold'
        )
        ax0.axis('off')
        if len(large_boxes) > 0 or len(medium_boxes) > 0:
            ax0.legend(loc='upper right', fontsize=8)
        
        # 子图2-4：掩码通道可视化
        mask_titles = ['Mask Channel 1\n(Large Box 1)', 'Mask Channel 2\n(Large Box 2)', 'Mask Channel 3\n(Medium Box)']
        mask_colors = ['Reds', 'Greens', 'Blues']
        
        for i in range(3):
            ax = axes[i + 1]
            mask = masks[i]  # [256, 256]
            
            # 可视化掩码（使用colormap）
            ax.imshow(mask, cmap=mask_colors[i], vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(mask_titles[i], fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # 显示掩码像素统计
            mask_pixels = mask.sum()
            ax.text(0.5, -0.1, f"Pixels: {mask_pixels}", 
                   transform=ax.transAxes, ha='center', fontsize=9)
        
        plt.suptitle(
            f"Bbox Extraction Result - {Path(json_path).stem}",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result
    
    def batch_visualize(
        self,
        json_dir: str,
        output_dir: str,
        num_visualize: int = 5,
        start_index: int = 0
    ):
        """
        批量可视化文件
        
        Args:
            json_dir: JSON文件目录
            output_dir: 输出目录
            num_visualize: 要可视化的文件数量
            start_index: 起始索引（跳过前面的文件）
        """
        json_dir = Path(json_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = sorted([f for f in json_dir.glob("sa_*.json")])
        
        # 从指定索引开始选择
        selected_files = json_files[start_index:start_index + num_visualize]
        
        print(f"\n📁 可视化 {len(selected_files)} 个文件（从第 {start_index+1} 个开始）...")
        
        for json_file in tqdm(selected_files, desc="可视化"):
            image_file = json_dir / f"{json_file.stem}.jpg"
            
            if not image_file.exists():
                print(f"⚠️  跳过 {json_file.stem}: 图像文件不存在")
                continue
            
            output_path = output_dir / f"{json_file.stem}_bboxes.png"
            
            try:
                result = self.visualize_boxes_on_image(
                    str(image_file),
                    str(json_file),
                    str(output_path)
                )
                
                # 保存掩码为.npy文件
                masks_dir = output_dir / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                mask_path = masks_dir / f"{json_file.stem}_masks.npy"
                np.save(str(mask_path), result['masks'])
                
                print(f"✅ {json_file.stem}: Large={len(result['large_boxes'])}, Medium={len(result['medium_boxes'])}, Mask shape: {result['masks'].shape}")
            except Exception as e:
                print(f"❌ {json_file.stem}: {e}")


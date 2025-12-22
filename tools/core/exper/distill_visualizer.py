#!/usr/bin/env python3
"""
独立的可视化模块：用于蒸馏训练过程的多维度可视化
包含特征对齐、边缘生成、掩码辨析、损失权重等可视化功能
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


class DistillVisualizer:
    """蒸馏训练可视化器"""
    
    def __init__(self, output_dir, config, device):
        """
        Args:
            output_dir: 输出根目录
            config: 训练配置字典
            device: torch.device
        """
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = device
        
        # 预定义均值和标准差用于反归一化 (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def denormalize(self, images):
        """
        将归一化的 Tensor 转回 [0, 255] 的 numpy array (H, W, C)
        
        Args:
            images: [B, 3, H, W] float32 tensor (ImageNet normalized)
            
        Returns:
            [B, H, W, 3] uint8 numpy array
        """
        with torch.no_grad():
            img = images * self.std + self.mean
            img = torch.clamp(img, 0, 1) * 255.0
            img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return img

    def get_feature_heatmap(self, feature, size=(256, 256)):
        """
        计算特征的平均激活热力图
        
        Args:
            feature: [B, C, H, W] tensor
            size: 目标尺寸 (H, W)
            
        Returns:
            [B, H, W] numpy array (如果是单个样本) 或 [H, W] numpy array
        """
        if feature is None:
            return np.zeros(size, dtype=np.float32)
        with torch.no_grad():
            # 计算通道维度的均值
            heatmap = torch.mean(torch.abs(feature), dim=1, keepdim=True)  # [B, 1, H, W]
            heatmap = F.interpolate(heatmap, size=size, mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze().cpu().numpy()  # [B, H, W] or [H, W] if B=1
        return heatmap

    def compute_loss_weights_map(self, fg_map, bg_map, edge_map=None):
        """
        根据配置复现损失函数的空间权重图 (FGD/FSD) 用于可视化
        
        Args:
            fg_map: [B, 1, H, W] 前景掩码
            bg_map: [B, 1, H, W] 背景掩码
            edge_map: [B, 1, H, W] 边缘掩码 (可选)
            
        Returns:
            [B, 1, H, W] tensor (归一化到 0-1)
        """
        loss_type = self.config.get("loss_type", "mse")
        weight_map = torch.ones_like(fg_map)

        if loss_type == "fgd":
            alpha = self.config.get("fgd_alpha_fg", 0.001)
            beta = self.config.get("fgd_beta_bg", 0.0005)
            
            # 【修复】FGD损失实际使用的是 sqrt(fg_map) 和 sqrt(bg_map) 作为空间权重
            # 然后 alpha_fg 和 beta_bg 是在损失值级别加权的
            # 可视化时显示每个像素的实际权重贡献：sqrt(fg_map) * alpha_fg + sqrt(bg_map) * beta_bg
            sqrt_fg = torch.sqrt(torch.clamp(fg_map, min=0))
            sqrt_bg = torch.sqrt(torch.clamp(bg_map, min=0))
            weight_map = sqrt_fg * alpha + sqrt_bg * beta
            
            # 边缘增强（如果启用）
            enable_edge_boost = self.config.get("enable_edge_boost", False)
            if enable_edge_boost and edge_map is not None:
                alpha_edge = self.config.get("fgd_alpha_edge", 0.002)
                # 边缘区域也使用 sqrt(edge_map) 作为空间权重
                sqrt_edge = torch.sqrt(torch.clamp(edge_map, min=0))
                # 边缘损失是独立计算的，但在可视化中叠加显示其权重贡献
                weight_map = weight_map + sqrt_edge * alpha_edge

        elif loss_type == "fsdlike":
            w_fg = self.config.get("fsd_weight_fg", 1.0)
            w_bg = self.config.get("fsd_weight_bg", 0.2)
            weight_map = fg_map * w_fg + bg_map * w_bg
        
        # 归一化到 0-1 以便可视化 (Min-Max)
        with torch.no_grad():
            w_min = weight_map.min()
            w_max = weight_map.max()
            if w_max > w_min:
                weight_map = (weight_map - w_min) / (w_max - w_min)

        return weight_map

    def draw_bboxes(self, img_np, bboxes, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制边界框
        
        Args:
            img_np: [H, W, 3] uint8 numpy array
            bboxes: [N, 4] tensor 或 numpy array (XYXY格式) 或 None
            color: BGR颜色元组
            thickness: 线条粗细
            
        Returns:
            [H, W, 3] uint8 numpy array (绘制了bbox的图像)
        """
        img_copy = img_np.copy()
        if bboxes is None or (isinstance(bboxes, torch.Tensor) and bboxes.numel() == 0):
            return img_copy
        
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(bboxes, np.ndarray) and bboxes.size == 0:
            return img_copy
            
        for box in bboxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_copy.shape[1] - 1))
            y1 = max(0, min(y1, img_copy.shape[0] - 1))
            x2 = max(0, min(x2, img_copy.shape[1] - 1))
            y2 = max(0, min(y2, img_copy.shape[0] - 1))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        return img_copy

    def visualize(self, 
                  batch, 
                  student_features, 
                  student_edge_logits, 
                  student_mask_logits, 
                  epoch, 
                  batch_idx, 
                  max_samples=4):
        """
        核心可视化函数
        
        Args:
            batch: DataLoader 取出的 batch 字典
            student_features: 字典 {'s16': tensor, 's32': tensor}，s16是已对齐的特征
            student_edge_logits: 学生边缘头输出 [B, 1, 256, 256] 或 None
            student_mask_logits: 学生 Mask Decoder 输出 [B, 1, 256, 256] 或 None
            epoch: 当前epoch
            batch_idx: 当前batch索引
            max_samples: 最多可视化几个样本
        """
        images = batch["image"]  # 可能是 uint8 或 float32，不需要移动到设备（后面直接转换）
        teacher_feat = batch["teacher_features"].to(self.device)
        edge_gt = batch["edge_256x256"].to(self.device)
        
        # 取部分样本进行可视化
        B = images.shape[0]
        N = min(B, max_samples)
        
        # 准备数据：处理 uint8 图像（batch["image"] 通常是 uint8 CHW 格式）
        with torch.no_grad():
            if images.dtype == torch.uint8:
                # uint8 格式：直接转换为 numpy
                imgs_np = images[:N].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            else:
                # float32 格式（归一化后的）：需要反归一化
                imgs_np = self.denormalize(images[:N].to(self.device))
        
        # 特征热力图
        s16 = student_features.get('s16')  # 已对齐的student特征
        s32 = student_features.get('s32')
        
        # 对齐 Student P4 (S16) 到 Teacher 尺寸
        if s16.shape[-2:] != teacher_feat.shape[-2:]:
            s16_resized = F.interpolate(s16[:N], size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
        else:
            s16_resized = s16[:N]

        # 热力图计算
        heatmap_s16 = self.get_feature_heatmap(s16_resized)  # Student P4
        heatmap_t16 = self.get_feature_heatmap(teacher_feat[:N])  # Teacher P4
        heatmap_s32 = []
        if s32 is not None:
            for i in range(N):
                hm = self.get_feature_heatmap(s32[i:i+1])
                heatmap_s32.append(hm.squeeze() if hm.ndim > 2 else hm)
        else:
            heatmap_s32 = [None] * N
        
        # 权重图计算 (可视化 Loss 关注哪里)
        fg_map = batch.get("fg_map")
        bg_map = batch.get("bg_map")
        if fg_map is None:
            fg_map = torch.zeros_like(edge_gt)
        if bg_map is None:
            bg_map = torch.zeros_like(edge_gt)
        fg_map = fg_map[:N].to(self.device)
        bg_map = bg_map[:N].to(self.device)
        
        edge_map_w = batch.get("edge_map")
        if edge_map_w is not None:
            if isinstance(edge_map_w, torch.Tensor):
                if edge_map_w.numel() > 0:
                    edge_map_w = edge_map_w[:N].to(self.device)
                else:
                    edge_map_w = None
            else:
                edge_map_w = None
        
        weight_maps = self.compute_loss_weights_map(fg_map, bg_map, edge_map_w)
        
        # 绘图配置
        rows = 4
        cols = 5
        fig_size = (cols * 4, rows * 4)
        
        for i in range(N):
            fig, axes = plt.subplots(rows, cols, figsize=fig_size)
            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            fig.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Sample {i} | ID: {batch['image_id'][i] if 'image_id' in batch else 'unknown'}", fontsize=16)
            
            # --- Row 1: 基础图像信息 ---
            # 1.1 原图
            ax = axes[0, 0]
            ax.imshow(imgs_np[i])
            ax.set_title("Original Image (1024x1024)")
            ax.axis('off')
            
            # 1.2 原图 + GT Box
            ax = axes[0, 1]
            # 获取当前样本的 box
            boxes = None
            if "box_prompts_xyxy" in batch:
                raw_box = batch["box_prompts_xyxy"][i]
                if isinstance(raw_box, torch.Tensor):
                    if raw_box.numel() > 0:
                        boxes = raw_box
                elif isinstance(raw_box, np.ndarray):
                    if raw_box.size > 0:
                        boxes = torch.from_numpy(raw_box)
            
            img_bbox = self.draw_bboxes(imgs_np[i], boxes)
            ax.imshow(img_bbox)
            num_boxes = len(boxes) if boxes is not None and (isinstance(boxes, torch.Tensor) or isinstance(boxes, np.ndarray)) and len(boxes.shape) > 0 else 0
            ax.set_title(f"Image + GT Boxes (N={num_boxes})")
            ax.axis('off')

            # 1.3 前景图 (FG Map)
            ax = axes[0, 2]
            fg_np = fg_map[i].squeeze().cpu().numpy()
            ax.imshow(fg_np, cmap='gray')
            ax.set_title("Foreground Map")
            ax.axis('off')

            # 1.4 背景图 (BG Map)
            ax = axes[0, 3]
            bg_np = bg_map[i].squeeze().cpu().numpy()
            ax.imshow(bg_np, cmap='gray')
            ax.set_title("Background Map")
            ax.axis('off')
            
            # 1.5 损失权重图 (FGD/FSD Weights)
            ax = axes[0, 4]
            wm = weight_maps[i].squeeze().cpu().numpy()
            im = ax.imshow(wm, cmap='jet')
            # 标注这是FGD损失中的权重（包含边缘增强）
            loss_type = self.config.get("loss_type", "mse")
            enable_edge_boost = self.config.get("enable_edge_boost", False)
            if loss_type == "fgd" and enable_edge_boost:
                title_str = "FGD Loss Weights\n(Edge Boost: ON)"
            elif loss_type == "fgd":
                title_str = "FGD Loss Weights\n(Edge Boost: OFF)"
            else:
                title_str = f"{loss_type.upper()} Loss Weights"
            ax.set_title(title_str)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')

            # --- Row 2: 边缘检测 (Edge Task) ---
            # 2.1 Teacher Edge (GT)
            ax = axes[1, 0]
            edge_gt_np = edge_gt[i].squeeze().cpu().numpy()
            ax.imshow(edge_gt_np, cmap='gray')
            ax.set_title("Teacher Edge (GT)")
            ax.axis('off')

            # 2.2 Student Edge (Pred)
            ax = axes[1, 1]
            if student_edge_logits is not None:
                edge_pred = torch.sigmoid(student_edge_logits[i]).detach().cpu().numpy().squeeze()
                ax.imshow(edge_pred, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"Student Edge Pred\n(Max: {edge_pred.max():.2f})")
            else:
                ax.text(0.5, 0.5, "Edge Head Disabled", ha='center', va='center')
                ax.set_title("Student Edge Pred")
            ax.axis('off')

            # 2.3 Edge Overlay on Image
            ax = axes[1, 2]
            if student_edge_logits is not None:
                edge_pred = torch.sigmoid(student_edge_logits[i]).detach().cpu().numpy().squeeze()
                # 简单叠加红色边缘
                edge_mask = (edge_pred > 0.3).astype(np.float32)
                overlay = imgs_np[i].copy()
                # Resize edge to 1024
                edge_big = cv2.resize(edge_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # 创建彩色边缘叠加
                overlay_float = overlay.astype(np.float32)
                edge_big_3d = np.stack([edge_big] * 3, axis=-1)
                overlay_float = overlay_float * (1 - edge_big_3d * 0.5) + np.array([255, 0, 0]) * (edge_big_3d * 0.5)
                overlay = overlay_float.astype(np.uint8)
                ax.imshow(overlay)
                ax.set_title("Edge Overlay (Pred)")
            else:
                ax.text(0.5, 0.5, "Edge Overlay N/A", ha='center', va='center')
            ax.axis('off')
            
            # 2.4 Edge Difference
            ax = axes[1, 3]
            if student_edge_logits is not None:
                edge_pred = torch.sigmoid(student_edge_logits[i]).detach().cpu().numpy().squeeze()
                edge_diff = np.abs(edge_pred - edge_gt_np)
                im = ax.imshow(edge_diff, cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"Edge Error\n(MAE: {edge_diff.mean():.3f})")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "Edge Error N/A", ha='center', va='center')
            ax.axis('off')

            # 2.5 空白占位
            axes[1, 4].axis('off')

            # --- Row 3: 特征对齐 (Features P4/S16) ---
            # 3.1 Teacher Feature Heatmap
            ax = axes[2, 0]
            hm_t = heatmap_t16[i] if len(heatmap_t16.shape) == 3 else heatmap_t16
            im = ax.imshow(hm_t, cmap='viridis')
            ax.set_title("Teacher P4 (S16) Mean")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')

            # 3.2 Student Feature Heatmap
            ax = axes[2, 1]
            hm_s = heatmap_s16[i] if len(heatmap_s16.shape) == 3 else heatmap_s16
            im = ax.imshow(hm_s, cmap='viridis')
            ax.set_title("Student P4 (S16) Mean")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')

            # 3.3 Feature Difference
            ax = axes[2, 2]
            # 简单的 L1 差异
            if s16_resized[i].shape == teacher_feat[i].shape:
                diff = torch.abs(s16_resized[i] - teacher_feat[i]).mean(dim=0).cpu().numpy()
                im = ax.imshow(diff, cmap='magma')
                ax.set_title(f"Diff (Student - Teacher)\n(MAE: {diff.mean():.3f})")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "Shape Mismatch", ha='center', va='center')
            ax.axis('off')
            
            # 3.4 Student P5 (S32) Heatmap
            ax = axes[2, 3]
            if heatmap_s32[i] is not None:
                im = ax.imshow(heatmap_s32[i], cmap='viridis')
                ax.set_title("Student P5 (S32) Mean")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "P5 Not Available", ha='center', va='center')
                ax.set_title("Student P5 (S32)")
            ax.axis('off')
            
            # 3.5 Teacher P5 (Missing)
            axes[2, 4].text(0.5, 0.5, "Teacher P5 N/A\n(Only S16 in NPZ)", ha='center', va='center')
            axes[2, 4].axis('off')

            # --- Row 4: 掩码辨析 (Mask Task) ---
            # 4.1 Teacher Mask (GT)
            ax = axes[3, 0]
            mask_gt = None
            if "box_prompts_masks_orig" in batch:
                raw_mask = batch["box_prompts_masks_orig"][i]
                if isinstance(raw_mask, torch.Tensor):
                    if raw_mask.numel() > 0:
                        # 通常是 [1, 256, 256] 或 [256, 256]
                        mask_gt = raw_mask.cpu().numpy().squeeze()
                elif isinstance(raw_mask, np.ndarray):
                    if raw_mask.size > 0:
                        if raw_mask.dtype == object and raw_mask.ndim == 0:
                            mask_gt = raw_mask.item()  # handle object array
                        else:
                            mask_gt = raw_mask
            
            if mask_gt is not None and mask_gt.size > 0:
                if isinstance(mask_gt, np.ndarray):
                    ax.imshow(mask_gt, cmap='gray', vmin=0, vmax=1)
                    ax.set_title("Teacher Mask GT (256x256)")
                else:
                    ax.text(0.5, 0.5, "Invalid Mask", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No GT Mask", ha='center', va='center')
                ax.set_title("Teacher Mask GT")
            ax.axis('off')

            # 4.2 Student Mask Logits (Pred)
            ax = axes[3, 1]
            if student_mask_logits is not None:
                # Logits -> Sigmoid
                m_pred = torch.sigmoid(student_mask_logits[i]).detach().cpu().numpy().squeeze()
                im = ax.imshow(m_pred, cmap='plasma', vmin=0, vmax=1)
                ax.set_title("Student Mask Pred (SAM2)")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "Mask Head Disabled", ha='center', va='center')
                ax.set_title("Student Mask Pred")
            ax.axis('off')
            
            # 4.3 Mask Overlay
            ax = axes[3, 2]
            if student_mask_logits is not None:
                m_pred = torch.sigmoid(student_mask_logits[i]).detach().cpu().numpy().squeeze()
                m_bin = (m_pred > 0.5).astype(np.float32)
                overlay_m = imgs_np[i].copy()
                m_big = cv2.resize(m_bin, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # Green overlay for mask
                overlay_m_float = overlay_m.astype(np.float32)
                m_big_3d = np.stack([m_big] * 3, axis=-1)
                overlay_m_float = overlay_m_float * (1 - m_big_3d * 0.5) + np.array([0, 255, 0]) * (m_big_3d * 0.5)
                overlay_m = overlay_m_float.astype(np.uint8)
                ax.imshow(overlay_m)
                ax.set_title("Mask Pred Overlay")
            else:
                ax.text(0.5, 0.5, "Mask Overlay N/A", ha='center', va='center')
            ax.axis('off')

            # 4.4 Mask Difference
            ax = axes[3, 3]
            if student_mask_logits is not None and mask_gt is not None and isinstance(mask_gt, np.ndarray) and mask_gt.size > 0:
                m_pred = torch.sigmoid(student_mask_logits[i]).detach().cpu().numpy().squeeze()
                # 确保mask_gt是256x256
                if mask_gt.shape != (256, 256):
                    mask_gt_resized = cv2.resize(mask_gt.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_gt_resized = mask_gt.astype(np.float32)
                mask_diff = np.abs(m_pred - mask_gt_resized)
                im = ax.imshow(mask_diff, cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"Mask Error\n(MAE: {mask_diff.mean():.3f})")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "Mask Error N/A", ha='center', va='center')
            ax.axis('off')

            # 4.5 空白占位
            axes[3, 4].axis('off')

            # 保存
            save_path = self.output_dir / f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        # 为了不刷屏日志，只在第一个样本打印一次
        print(f"[VIS] Saved visualization batch to {self.output_dir}")

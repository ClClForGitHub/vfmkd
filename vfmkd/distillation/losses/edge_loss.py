"""
边缘蒸馏损失函数
实现BCE + DICE损失用于边缘检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice损失函数
    用于处理边缘检测中的类别不平衡问题
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: 平滑参数，避免除零
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        
        Args:
            pred: 预测概率 [B, H, W] 或 [B, 1, H, W]
            target: 真值标签 [B, H, W] 或 [B, 1, H, W]
            
        Returns:
            Dice损失值
        """
        # 确保输入形状一致
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        
        # 将预测转换为概率
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回Dice损失 (1 - Dice)
        return 1.0 - dice.mean()


class EdgeDistillationLoss(nn.Module):
    """
    边缘蒸馏损失函数
    结合BCE和DICE损失，用于边缘检测蒸馏
    支持渐进式边缘区域掩码（前N个epoch全局学习，之后只在膨胀边缘区域内计算损失）
    """
    
    def __init__(self, 
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 dice_smooth: float = 1e-6,
                 enable_edge_mask: bool = False,
                 edge_mask_kernel_size: int = 3,
                 use_pos_weight: bool = False):
        """
        Args:
            bce_weight: BCE损失权重
            dice_weight: DICE损失权重
            dice_smooth: DICE损失平滑参数
            enable_edge_mask: 是否启用边缘区域掩码（只在膨胀后的边缘区域内计算损失）
            edge_mask_kernel_size: 边缘膨胀核大小（建议3，即膨胀1像素）
            use_pos_weight: 是否启用正类重加权（自动计算负正比作为pos_weight）
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.enable_edge_mask = enable_edge_mask
        self.use_pos_weight = use_pos_weight
        
        # 损失函数
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # 改为none，手动应用mask
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        
        # 边缘膨胀器（使用MaxPool2d实现高效膨胀）
        self.edge_dilater = nn.MaxPool2d(
            kernel_size=edge_mask_kernel_size,
            stride=1,
            padding=edge_mask_kernel_size // 2
        )
    
    def forward(self, 
                student_pred: torch.Tensor,
                teacher_target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        计算边缘蒸馏损失
        
        Args:
            student_pred: 学生模型预测logits [B, 1, H, W] 或 [B, H, W]
            teacher_target: 教师边缘真值 [B, H, W] 或 [B, 1, H, W]
            return_components: 是否返回各个损失组件
            
        Returns:
            总损失值，如果return_components=True则返回字典
        """
        # 确保输入形状一致
        original_shape = student_pred.shape
        if student_pred.dim() == 4 and student_pred.size(1) == 1:
            student_pred = student_pred.squeeze(1)
        if teacher_target.dim() == 4 and teacher_target.size(1) == 1:
            teacher_target = teacher_target.squeeze(1)
        
        # 生成边缘区域掩码（如果启用）
        edge_mask = None
        if self.enable_edge_mask:
            with torch.no_grad():
                # 需要4D张量进行MaxPool2d操作
                if teacher_target.dim() == 3:
                    teacher_target_4d = teacher_target.unsqueeze(1)  # [B, 1, H, W]
                else:
                    teacher_target_4d = teacher_target
                
                # 使用MaxPool2d进行高效膨胀（3x3核，膨胀1像素）
                edge_mask = self.edge_dilater(teacher_target_4d)  # [B, 1, H, W]
                edge_mask = edge_mask.squeeze(1)  # [B, H, W]
        
        # 计算BCE损失（带掩码和正类重加权）
        bce_loss_per_pixel = self.bce_loss(student_pred, teacher_target)  # [B, H, W]
        
        # 正类重加权
        pos_weight_value = 1.0
        if self.use_pos_weight:
            with torch.no_grad():
                # 选择统计区域：如果有边带掩码则在边带内统计，否则全局统计
                stat_region = edge_mask if edge_mask is not None else torch.ones_like(teacher_target)
                
                # 统计正负样本数量
                pos_mask = (teacher_target > 0.5).float() * stat_region
                neg_mask = (teacher_target <= 0.5).float() * stat_region
                
                pos_count = pos_mask.sum().clamp(min=1.0)
                neg_count = neg_mask.sum().clamp(min=1.0)
                
                # 计算正类权重：负正比，限制在[1.0, 10.0]范围内
                pos_weight_value = (neg_count / pos_count).clamp(min=1.0, max=10.0).item()
            
            # 应用正类重加权
            sample_weight = torch.where(
                teacher_target > 0.5,
                torch.full_like(teacher_target, pos_weight_value),
                torch.ones_like(teacher_target)
            )
            bce_loss_per_pixel = bce_loss_per_pixel * sample_weight
        
        if edge_mask is not None:
            # 只在边缘区域内计算损失
            bce_loss_masked = bce_loss_per_pixel * edge_mask
            # 归一化：除以有效像素数量
            num_valid_pixels = edge_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            bce_loss = (bce_loss_masked.sum(dim=(1, 2), keepdim=True) / num_valid_pixels).mean()
        else:
            # 全局计算损失
            bce_loss = bce_loss_per_pixel.mean()
        
        # 计算DICE损失（带掩码）
        if edge_mask is not None:
            # 为Dice损失也应用掩码
            dice_loss = self._masked_dice_loss(student_pred, teacher_target, edge_mask)
        else:
            dice_loss = self.dice_loss(student_pred, teacher_target)
        
        # 总损失
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        if return_components:
            result = {
                'total_loss': total_loss,
                'bce_loss': bce_loss,
                'dice_loss': dice_loss,
                'bce_weight': self.bce_weight,
                'dice_weight': self.dice_weight
            }
            if edge_mask is not None:
                result['edge_mask_ratio'] = edge_mask.mean().item()  # 掩码覆盖率
            if self.use_pos_weight:
                result['pos_weight'] = pos_weight_value  # 正类权重
            return result
        else:
            return total_loss
    
    def _masked_dice_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算带掩码的Dice损失（只在掩码区域内计算）
        
        Args:
            pred: 预测logits [B, H, W]
            target: 真值标签 [B, H, W]
            mask: 边缘区域掩码 [B, H, W]
            
        Returns:
            掩码Dice损失
        """
        # 将预测转换为概率
        pred = torch.sigmoid(pred)
        
        # 应用掩码
        pred_masked = pred * mask
        target_masked = target * mask
        
        # 展平
        pred_flat = pred_masked.view(pred_masked.size(0), -1)
        target_flat = target_masked.view(target_masked.size(0), -1)
        mask_flat = mask.view(mask.size(0), -1)
        
        # 计算交集和并集（在掩码区域内）
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # 计算Dice系数
        smooth = self.dice_loss.smooth
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # 返回Dice损失 (1 - Dice)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal损失函数
    用于处理边缘检测中的难易样本不平衡问题
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: 平衡正负样本的权重
            gamma: 难易样本的权重
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Focal损失
        
        Args:
            pred: 预测logits [B, H, W] 或 [B, 1, H, W]
            target: 真值标签 [B, H, W] 或 [B, 1, H, W]
            
        Returns:
            Focal损失值
        """
        # 确保输入形状一致
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算Focal权重
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 计算Focal损失
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class EdgeDistillationLossAdvanced(nn.Module):
    """
    高级边缘蒸馏损失函数
    结合BCE、DICE和Focal损失
    """
    
    def __init__(self,
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 dice_smooth: float = 1e-6):
        """
        Args:
            bce_weight: BCE损失权重
            dice_weight: DICE损失权重
            focal_weight: Focal损失权重
            focal_alpha: Focal损失alpha参数
            focal_gamma: Focal损失gamma参数
            dice_smooth: DICE损失平滑参数
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # 损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self,
                student_pred: torch.Tensor,
                teacher_target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        计算高级边缘蒸馏损失
        
        Args:
            student_pred: 学生模型预测logits
            teacher_target: 教师边缘真值
            return_components: 是否返回各个损失组件
            
        Returns:
            总损失值或损失组件字典
        """
        # 确保输入形状一致
        if student_pred.dim() == 4 and student_pred.size(1) == 1:
            student_pred = student_pred.squeeze(1)
        if teacher_target.dim() == 4 and teacher_target.size(1) == 1:
            teacher_target = teacher_target.squeeze(1)
        
        # 计算各种损失
        bce_loss = self.bce_loss(student_pred, teacher_target)
        dice_loss = self.dice_loss(student_pred, teacher_target)
        
        losses = {
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'bce_weight': self.bce_weight,
            'dice_weight': self.dice_weight
        }
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # 可选的Focal损失
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(student_pred, teacher_target)
            losses['focal_loss'] = focal_loss
            losses['focal_weight'] = self.focal_weight
            total_loss += self.focal_weight * focal_loss
        
        losses['total_loss'] = total_loss
        
        if return_components:
            return losses
        else:
            return total_loss


def test_edge_loss():
    """测试边缘损失函数"""
    print("=== 测试边缘损失函数 ===")
    
    # 创建测试数据
    batch_size = 2
    height, width = 256, 256
    
    # 学生预测logits
    student_pred = torch.randn(batch_size, height, width)
    
    # 教师边缘真值
    teacher_target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    print(f"学生预测形状: {student_pred.shape}")
    print(f"教师真值形状: {teacher_target.shape}")
    
    # 测试基础边缘蒸馏损失
    print("\n1. 测试EdgeDistillationLoss")
    edge_loss = EdgeDistillationLoss(bce_weight=0.5, dice_weight=0.5)
    
    loss_dict = edge_loss(student_pred, teacher_target, return_components=True)
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"BCE损失: {loss_dict['bce_loss'].item():.4f}")
    print(f"DICE损失: {loss_dict['dice_loss'].item():.4f}")
    
    # 测试高级边缘蒸馏损失
    print("\n2. 测试EdgeDistillationLossAdvanced")
    advanced_loss = EdgeDistillationLossAdvanced(
        bce_weight=0.5,
        dice_weight=0.5,
        focal_weight=0.5
    )
    
    loss_dict = advanced_loss(student_pred, teacher_target, return_components=True)
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"BCE损失: {loss_dict['bce_loss'].item():.4f}")
    print(f"DICE损失: {loss_dict['dice_loss'].item():.4f}")
    print(f"Focal损失: {loss_dict['focal_loss'].item():.4f}")
    
    # 测试梯度
    print("\n3. 测试梯度计算")
    student_pred.requires_grad_(True)
    loss = edge_loss(student_pred, teacher_target)
    loss.backward()
    
    print(f"梯度形状: {student_pred.grad.shape}")
    print(f"梯度范围: [{student_pred.grad.min().item():.4f}, {student_pred.grad.max().item():.4f}]")
    
    print("\n✅ 边缘损失函数测试完成！")


if __name__ == "__main__":
    test_edge_loss()

"""
特征对齐适配器
用于将学生特征对齐到教师特征的通道数
只处理通道对齐，同分辨率只需要1x1卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ChannelAdapter(nn.Module):
    """
    通道对齐适配器
    只处理通道对齐，使用1x1卷积
    """
    
    def __init__(self, student_channels: int, teacher_channels: int):
        """
        初始化通道适配器
        
        Args:
            student_channels: 学生特征通道数
            teacher_channels: 教师特征通道数
        """
        super().__init__()
        
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        
        # 只有通道数不同时才需要对齐
        if student_channels != teacher_channels:
            self.adapter = nn.Conv2d(
                student_channels, 
                teacher_channels, 
                kernel_size=1, 
                bias=False
            )
            # 初始化权重
            nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        else:
            self.adapter = None
    
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，对齐学生特征通道
        
        Args:
            student_features: 学生特征 [B, C_student, H, W]
            
        Returns:
            对齐后的特征 [B, C_teacher, H, W]
        """
        if self.adapter is not None:
            return self.adapter(student_features)
        else:
            return student_features


class SimpleAdapter(nn.Module):
    """
    简单特征对齐适配器
    根据输入特征自动创建对齐层
    """
    
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict()
    
    def _get_adapter_key(self, student_channels: int, teacher_channels: int) -> str:
        """生成适配器键"""
        return f"{student_channels}to{teacher_channels}"
    
    def forward(self, 
                student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，自动对齐特征通道和空间尺寸
        
        Args:
            student_features: 学生特征
            teacher_features: 教师特征（用于确定目标通道数和空间尺寸）
            
        Returns:
            对齐后的学生特征
        """
        student_channels = student_features.shape[1]
        teacher_channels = teacher_features.shape[1]
        
        # 先进行通道对齐
        if student_channels != teacher_channels:
            adapter_key = self._get_adapter_key(student_channels, teacher_channels)
            
            if adapter_key not in self.adapters:
                adapter = ChannelAdapter(student_channels, teacher_channels)
                self.adapters[adapter_key] = adapter.to(student_features.device)
            
            student_features = self.adapters[adapter_key](student_features)
        
        # 再进行空间对齐（如果需要）
        if student_features.shape[2:] != teacher_features.shape[2:]:
            # 使用双线性插值对齐空间尺寸
            student_features = F.interpolate(
                student_features, 
                size=teacher_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return student_features

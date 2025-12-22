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
    
    def prepare_from_checkpoint(self, checkpoint_state_dict):
        """
        从checkpoint中解析适配器键，提前创建适配器
        
        这个方法的目的是：在load_state_dict之前创建适配器，确保权重能正确加载。
        之后在forward()中遇到相同通道组合时，不会重复创建（因为已有if检查）。
        
        Args:
            checkpoint_state_dict: checkpoint中的feature_adapter state_dict
        
        Returns:
            list: 创建的适配器键列表
        """
        adapter_keys = []
        for key in checkpoint_state_dict.keys():
            if key.startswith('adapters.') and '.adapter.weight' in key:
                # 格式: adapters.{student_ch}to{teacher_ch}.adapter.weight
                parts = key.split('.')
                if len(parts) >= 2:
                    adapter_key = parts[1]  # {student_ch}to{teacher_ch}
                    
                    # 关键：检查是否已存在，避免重复创建
                    if adapter_key not in self.adapters:
                        # 解析通道数
                        student_ch, teacher_ch = map(int, adapter_key.split('to'))
                        # 创建适配器（权重会在load_state_dict时加载）
                        adapter = ChannelAdapter(student_ch, teacher_ch)
                        # 获取设备：优先从checkpoint state_dict的权重位置获取设备
                        # 如果state_dict中有权重，使用其设备；否则尝试从已有参数获取
                        weight_key = f'adapters.{adapter_key}.adapter.weight'
                        if weight_key in checkpoint_state_dict:
                            device = checkpoint_state_dict[weight_key].device
                        elif list(self.parameters()):
                            device = next(self.parameters()).device
                        else:
                            device = torch.device('cpu')
                        self.adapters[adapter_key] = adapter.to(device)
                        adapter_keys.append(adapter_key)
        
        if adapter_keys:
            print(f"[Adapter] 从checkpoint预创建适配器: {adapter_keys}")
        elif any(k.startswith('adapters.') for k in checkpoint_state_dict.keys()):
            # 如果有适配器键但没创建新适配器，说明都已存在
            existing_keys = [k.split('.')[1] for k in checkpoint_state_dict.keys() 
                           if k.startswith('adapters.') and '.adapter.weight' in k]
            print(f"[Adapter] 适配器已存在，无需创建: {existing_keys}")
        
        return adapter_keys
    
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


class EdgeAdapter(nn.Module):
    """
    边缘特征对齐适配器
    专门用于边缘检测：将backbone的S4特征对齐到256通道、256×256
    设计目标：固定输出256通道、256×256，代码保证输入是256×256
    """
    
    def __init__(self, target_channels: int = 256, target_size: int = 256):
        """
        初始化边缘适配器
        
        Args:
            target_channels: 目标通道数（默认256）
            target_size: 目标空间尺寸（默认256×256）
        """
        super().__init__()
        self.target_channels = target_channels
        self.target_size = target_size
        self.adapters = nn.ModuleDict()
    
    def _get_adapter_key(self, input_channels: int) -> str:
        """生成适配器键：{input_channels}to{target_channels}"""
        return f"{input_channels}to{self.target_channels}"
    
    def prepare_from_checkpoint(self, checkpoint_state_dict):
        """
        从checkpoint中解析适配器键，提前创建适配器
        
        这个方法的目的是：在load_state_dict之前创建适配器，确保权重能正确加载。
        
        Args:
            checkpoint_state_dict: checkpoint中的edge_adapter state_dict
        
        Returns:
            list: 创建的适配器键列表
        """
        adapter_keys = []
        for key in checkpoint_state_dict.keys():
            if key.startswith('adapters.') and '.adapter.weight' in key:
                # 格式: adapters.{input_ch}to{target_ch}.adapter.weight
                parts = key.split('.')
                if len(parts) >= 2:
                    adapter_key = parts[1]  # {input_ch}to{target_ch}
                    
                    # 检查是否已存在，避免重复创建
                    if adapter_key not in self.adapters:
                        # 解析通道数
                        input_ch, target_ch = map(int, adapter_key.split('to'))
                        
                        # 只处理目标通道匹配的适配器
                        if target_ch == self.target_channels:
                            # 创建适配器（权重会在load_state_dict时加载）
                            adapter = ChannelAdapter(input_ch, target_ch)
                            # 获取设备
                            weight_key = f'adapters.{adapter_key}.adapter.weight'
                            if weight_key in checkpoint_state_dict:
                                device = checkpoint_state_dict[weight_key].device
                            elif list(self.parameters()):
                                device = next(self.parameters()).device
                            else:
                                device = torch.device('cpu')
                            self.adapters[adapter_key] = adapter.to(device)
                            adapter_keys.append(adapter_key)
        
        if adapter_keys:
            print(f"[EdgeAdapter] 从checkpoint预创建适配器: {adapter_keys}")
        elif any(k.startswith('adapters.') for k in checkpoint_state_dict.keys()):
            # 如果有适配器键但没创建新适配器，说明都已存在
            existing_keys = [k.split('.')[1] for k in checkpoint_state_dict.keys() 
                           if k.startswith('adapters.') and '.adapter.weight' in k]
            print(f"[EdgeAdapter] 适配器已存在，无需创建: {existing_keys}")
        
        return adapter_keys
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，对齐特征到256通道、256×256
        
        Args:
            x: 输入特征 (B, C_in, H, W)
               注意：代码保证输入空间尺寸是256×256，这里只做通道对齐
        Returns:
            对齐后的特征 (B, 256, 256, 256)
        """
        input_channels = x.size(1)
        
        # 通道对齐
        if input_channels != self.target_channels:
            adapter_key = self._get_adapter_key(input_channels)
            
            if adapter_key not in self.adapters:
                adapter = ChannelAdapter(input_channels, self.target_channels)
                self.adapters[adapter_key] = adapter.to(x.device)
            
            x = self.adapters[adapter_key](x)
        
        # 空间对齐（如果需要，虽然默认输入应该是256×256）
        # 添加这个检查是为了代码的健壮性
        if x.shape[2:] != (self.target_size, self.target_size):
            x = F.interpolate(
                x, 
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            )
        
        return x


# ===== V2 静态适配器 =====

class SimpleAdapterStatic(nn.Module):
    """
    静态特征适配器：在初始化时确定输入/输出通道，支持空间尺寸对齐（类似 EdgeAdapterStatic）
    """

    def __init__(self, in_channels: int, out_channels: int, target_size: int | tuple[int, int] | None = None):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # target_size 可以是 None（不插值）、int（正方形）或 (H, W) 元组
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        if self.target_size is not None:
            if isinstance(self.target_size, int):
                target_hw = (self.target_size, self.target_size)
            else:
                target_hw = self.target_size
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x


class EdgeAdapterStatic(nn.Module):
    """
    静态边缘适配器：在初始化时确定输入通道，输出固定到 target_channels、空间到 target_size
    """

    def __init__(self, in_channels: int, target_channels: int = 64, target_size: int = 256):
        super().__init__()
        self.target_size = target_size
        self.adapter = nn.Conv2d(in_channels, target_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return x

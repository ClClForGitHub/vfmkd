"""
通用边缘头实现
支持动态通道对齐的SimpleEdgeHead和LightweightEdgeHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Union


def _pick_gn_groups(C: int):
    """选择GroupNorm的组数"""
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1


class SimpleEdgeHead(nn.Module):
    """
    轻量边缘头：从对齐后的特征预测边缘概率
    结构：1×1降维 → DW 3×3 → 1×1到1ch → Sigmoid
    
    输入：对齐后的特征 (B, core_channels, H, W)
    输出：边缘logits (B, 1, H, W)
    """
    def __init__(self, core_channels: int = 256, output_channels: int = 1, init_p: float = 0.05):
        super().__init__()
        self.core_channels = core_channels
        self.output_channels = output_channels
        
        # 1×1降维（减参数）
        self.proj1 = nn.Conv2d(core_channels, core_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(_pick_gn_groups(core_channels), core_channels)
        self.act1 = nn.SiLU(inplace=True)
        
        # Depthwise 3×3（局部边缘信息）
        self.dw = nn.Conv2d(core_channels, core_channels, kernel_size=3, padding=1,
                           groups=core_channels, bias=False)
        self.gn2 = nn.GroupNorm(_pick_gn_groups(core_channels), core_channels)
        self.act2 = nn.SiLU(inplace=True)
        
        # 1×1输出层
        self.out = nn.Conv2d(core_channels, output_channels, kernel_size=1)
        
        self._init_weights(init_p)
    
    def _init_weights(self, init_p):
        """
        初始化权重
        - Kaiming初始化卷积层
        - 输出层偏置设为logit(p0)，初始偏向"少边缘"
        """
        for m in [self.proj1, self.dw]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # 输出层：权重接近零，偏置设为小概率
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            p = max(min(init_p, 0.99), 1e-3)
            self.out.bias.fill_(math.log(p / (1 - p)))  # logit(p0)
    
    def forward(self, x):
        """
        Args:
            x: 对齐后的特征 (B, core_channels, H, W)
        Returns:
            edge_logits: 边缘logits (B, 1, H, W)，未sigmoid
        """
        x = self.act1(self.gn1(self.proj1(x)))
        x = self.act2(self.gn2(self.dw(x)))
        x = self.out(x)  # 返回logits，不做sigmoid
        return x


class LightweightEdgeHead(nn.Module):
    """
    轻量边缘头：使用标准3×3卷积
    结构：3×3 → BN → ReLU → 3×3 → BN → ReLU → 1×1输出
    
    输入：对齐后的特征 (B, core_channels, H, W)
    输出：边缘logits (B, 1, H, W)
    """
    def __init__(self, core_channels: int = 256, output_channels: int = 1, init_p: float = 0.05):
        super().__init__()
        self.core_channels = core_channels
        self.output_channels = output_channels
        
        # 轻量边缘检测网络
        self.edge_net = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(core_channels, core_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(core_channels),
            nn.ReLU(inplace=True),
            
            # 第二个卷积块
            nn.Conv2d(core_channels, core_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(core_channels),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(core_channels, output_channels, 1)
        )
        
        self._init_weights(init_p)
    
    def _init_weights(self, init_p):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 输出层偏置初始化
        output_layer = self.edge_net[-1]
        nn.init.zeros_(output_layer.weight)
        with torch.no_grad():
            p = max(min(init_p, 0.99), 1e-3)
            output_layer.bias.fill_(math.log(p / (1 - p)))
    
    def forward(self, x):
        """
        Args:
            x: 对齐后的特征 (B, core_channels, H, W)
        Returns:
            edge_logits: 边缘logits (B, 1, H, W)，未sigmoid
        """
        return self.edge_net(x)


class UniversalEdgeHead(nn.Module):
    """
    通用边缘头（简化版）：固定接收64通道输入
    通道对齐由外部的EdgeAdapter处理，这里只做边缘检测
    """
    def __init__(self, 
                 core_channels: int = 64, 
                 output_channels: int = 1,
                 head_type: str = "simple",
                 init_p: float = 0.05):
        super().__init__()
        self.core_channels = core_channels
        self.output_channels = output_channels
        self.head_type = head_type
        
        # 固定接收64通道输入（由EdgeAdapter保证）
        assert core_channels == 64, "Edge head只支持64通道输入，使用EdgeAdapter做通道对齐"
        
        # 只创建边缘检测核心，不再包含通道对齐
        if head_type == "simple":
            self.edge_core = SimpleEdgeHead(core_channels, output_channels, init_p)
        elif head_type == "lightweight":
            self.edge_core = LightweightEdgeHead(core_channels, output_channels, init_p)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, 64, H, W) - 必须是64通道，由EdgeAdapter保证
        Returns:
            edge_logits: 边缘logits (B, 1, H, W)
        """
        assert x.size(1) == 64, f"输入必须是64通道，当前为{x.size(1)}。请使用EdgeAdapter进行通道对齐"
        
        # 直接进行边缘检测
        edge_logits = self.edge_core(x)
        
        return edge_logits
    
    def get_stats(self):
        """获取统计信息"""
        core_params = sum(p.numel() for p in self.edge_core.parameters())
        
        return {
            'head_type': self.head_type,
            'core_channels': self.core_channels,
            'core_parameters': core_params,
            'total_parameters': core_params,
        }
    
    def freeze_core(self):
        """冻结核心参数"""
        for param in self.edge_core.parameters():
            param.requires_grad = False
        print("[EdgeHead] 核心参数已冻结")
    
    def unfreeze_core(self):
        """解冻核心参数"""
        for param in self.edge_core.parameters():
            param.requires_grad = True
        print("[EdgeHead] 核心参数已解冻")


class EdgeHeadManager:
    """
    边缘头管理器：统一管理多个边缘头
    """
    def __init__(self, config: Dict):
        self.config = config
        self.edge_heads = nn.ModuleDict()
        self._create_edge_heads()
    
    def _create_edge_heads(self):
        """根据配置创建边缘头"""
        head_configs = self.config.get('edge_heads', {})
        
        for head_name, head_config in head_configs.items():
            head_type = head_config.get('type', 'simple')
            core_channels = head_config.get('core_channels', 64)
            output_channels = head_config.get('output_channels', 1)
            init_p = head_config.get('init_p', 0.05)
            
            self.edge_heads[head_name] = UniversalEdgeHead(
                core_channels=core_channels,
                output_channels=output_channels,
                head_type=head_type,
                init_p=init_p
            )
    
    def forward(self, x, head_name: str = "default", backbone_name: str = "default"):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            head_name: 边缘头名称
            backbone_name: backbone名称
        Returns:
            edge_logits: 边缘logits (B, 1, H, W)
        """
        if head_name not in self.edge_heads:
            raise ValueError(f"Edge head '{head_name}' not found")
        
        return self.edge_heads[head_name](x, backbone_name)
    
    def get_all_stats(self):
        """获取所有边缘头的统计信息"""
        stats = {}
        for head_name, head in self.edge_heads.items():
            stats[head_name] = head.get_stats()
        return stats


def create_edge_head(config: Dict) -> UniversalEdgeHead:
    """工厂函数：创建边缘头"""
    head_type = config.get('type', 'simple')
    core_channels = config.get('core_channels', 64)
    output_channels = config.get('output_channels', 1)
    init_p = config.get('init_p', 0.05)
    
    return UniversalEdgeHead(
        core_channels=core_channels,
        output_channels=output_channels,
        head_type=head_type,
        init_p=init_p
    )


def test_edge_head():
    """测试边缘头功能"""
    print("=== 测试边缘头功能 ===")
    
    # 创建边缘头
    edge_head = UniversalEdgeHead(
        core_channels=64,
        output_channels=1,
        head_type="simple"
    )
    
    # 测试不同backbone的特征
    backbones = {
        "yolov8": torch.randn(2, 256, 64, 64),  # YOLOv8 P3特征
        "repvit": torch.randn(2, 256, 64, 64),  # RepViT P3特征
        "vit": torch.randn(2, 384, 32, 32),     # ViT特征
    }
    
    for backbone_name, features in backbones.items():
        print(f"\n测试 {backbone_name}:")
        print(f"  输入形状: {features.shape}")
        
        # 前向传播
        edge_logits = edge_head(features, backbone_name)
        print(f"  输出形状: {edge_logits.shape}")
        print(f"  输出范围: [{edge_logits.min().item():.4f}, {edge_logits.max().item():.4f}]")
        
        # 转换为概率
        edge_probs = torch.sigmoid(edge_logits)
        print(f"  概率范围: [{edge_probs.min().item():.4f}, {edge_probs.max().item():.4f}]")
    
    # 获取统计信息
    stats = edge_head.get_stats()
    print(f"\n统计信息:")
    print(f"  核心参数: {stats['core_parameters']:,}")
    print(f"  对齐参数: {stats['aligner_parameters']:,}")
    print(f"  总参数: {stats['total_parameters']:,}")
    
    print("\n边缘头测试完成！")


if __name__ == "__main__":
    test_edge_head()

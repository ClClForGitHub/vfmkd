#!/usr/bin/env python3
"""
智能权重加载测试脚本
根据不同的使用场景实现不同的加载策略
"""

import torch
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.edge_head import UniversalEdgeHead
from vfmkd.distillation.adapters import SimpleAdapter
from vfmkd.distillation.losses.edge_loss import EdgeDistillationLoss
from vfmkd.distillation.losses.feature_loss import FeatureLoss


class SmartWeightLoader:
    """智能权重加载器"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def load_for_pretrained_backbone(self, checkpoint_path, backbone_type='yolov8', 
                                   load_edge_head=True, load_adapters=False):
        """
        场景1: 加载预训练backbone权重 (新训练)
        - Backbone: 必须加载
        - Edge Head: 可选加载 (核心通用组件)
        - Adapters: 不加载 (动态创建)
        - Optimizer: 不加载 (新训练)
        """
        print(f"🔄 场景1: 预训练backbone权重加载")
        print(f"   - Backbone: ✅ 必须加载")
        print(f"   - Edge Head: {'✅ 加载' if load_edge_head else '❌ 跳过'}")
        print(f"   - Adapters: {'✅ 加载' if load_adapters else '❌ 跳过 (动态创建)'}")
        print(f"   - Optimizer: ❌ 跳过 (新训练)")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建backbone
        backbone = self._create_backbone(backbone_type)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"✅ Backbone权重加载成功")
        
        # 创建edge head (核心通用组件)
        edge_head = UniversalEdgeHead(
            core_channels=64, output_channels=1, head_type='simple', init_p=0.05
        ).to(self.device)
        
        if load_edge_head and 'edge_head_state_dict' in checkpoint:
            missing_keys, unexpected_keys = edge_head.load_state_dict(
                checkpoint['edge_head_state_dict'], strict=False
            )
            print(f"✅ Edge Head权重加载成功 (多余键: {len(unexpected_keys)})")
        else:
            print(f"ℹ️  Edge Head使用随机初始化")
        
        # 创建adapters (动态组件)
        feature_adapter = SimpleAdapter().to(self.device)
        if load_adapters and 'feature_adapter_state_dict' in checkpoint:
            missing_keys, unexpected_keys = feature_adapter.load_state_dict(
                checkpoint['feature_adapter_state_dict'], strict=False
            )
            print(f"✅ Feature Adapter权重加载成功 (多余键: {len(unexpected_keys)})")
        else:
            print(f"ℹ️  Feature Adapter使用随机初始化")
        
        return backbone, edge_head, feature_adapter
    
    def load_for_training_resume(self, checkpoint_path, backbone_type='yolov8'):
        """
        场景2: 训练中断恢复 (完全加载)
        - Backbone: 必须加载
        - Edge Head: 必须加载
        - Adapters: 必须加载
        - Optimizer: 必须加载
        """
        print(f"🔄 场景2: 训练中断恢复")
        print(f"   - Backbone: ✅ 必须加载")
        print(f"   - Edge Head: ✅ 必须加载")
        print(f"   - Adapters: ✅ 必须加载")
        print(f"   - Optimizer: ✅ 必须加载")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建所有组件
        backbone = self._create_backbone(backbone_type)
        edge_head = UniversalEdgeHead(
            core_channels=64, output_channels=1, head_type='simple', init_p=0.05
        ).to(self.device)
        feature_adapter = SimpleAdapter().to(self.device)
        
        # 加载所有权重
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"✅ Backbone权重加载成功")
        
        missing_keys, unexpected_keys = edge_head.load_state_dict(
            checkpoint['edge_head_state_dict'], strict=False
        )
        print(f"✅ Edge Head权重加载成功 (多余键: {len(unexpected_keys)})")
        
        missing_keys, unexpected_keys = feature_adapter.load_state_dict(
            checkpoint['feature_adapter_state_dict'], strict=False
        )
        print(f"✅ Feature Adapter权重加载成功 (多余键: {len(unexpected_keys)})")
        
        # 创建optimizer并加载状态
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(edge_head.parameters()) + 
            list(feature_adapter.parameters()), lr=0.001
        )
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Optimizer状态加载成功")
        
        return backbone, edge_head, feature_adapter, optimizer
    
    def load_for_backbone_switch(self, checkpoint_path, from_backbone='yolov8', to_backbone='repvit'):
        """
        场景3: 切换backbone (部分加载)
        - Backbone: 重新创建，不加载
        - Edge Head: 加载 (核心通用组件)
        - Adapters: 不加载 (输入通道变化)
        - Optimizer: 不加载 (新训练)
        """
        print(f"🔄 场景3: 切换backbone ({from_backbone} → {to_backbone})")
        print(f"   - Backbone: ❌ 跳过 (重新创建 {to_backbone})")
        print(f"   - Edge Head: ✅ 加载 (核心通用组件)")
        print(f"   - Adapters: ❌ 跳过 (输入通道变化)")
        print(f"   - Optimizer: ❌ 跳过 (新训练)")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建新的backbone (不加载权重)
        backbone = self._create_backbone(to_backbone)
        print(f"✅ 新Backbone创建成功 ({to_backbone})")
        
        # 加载edge head (核心通用组件)
        edge_head = UniversalEdgeHead(
            core_channels=64, output_channels=1, head_type='simple', init_p=0.05
        ).to(self.device)
        
        if 'edge_head_state_dict' in checkpoint:
            missing_keys, unexpected_keys = edge_head.load_state_dict(
                checkpoint['edge_head_state_dict'], strict=False
            )
            print(f"✅ Edge Head权重加载成功 (多余键: {len(unexpected_keys)})")
        else:
            print(f"ℹ️  Edge Head使用随机初始化")
        
        # 创建新的adapters (不加载权重)
        feature_adapter = SimpleAdapter().to(self.device)
        print(f"ℹ️  Feature Adapter使用随机初始化 (输入通道变化)")
        
        return backbone, edge_head, feature_adapter
    
    def _create_backbone(self, backbone_type):
        """创建backbone"""
        if backbone_type == 'yolov8':
            config = {
                'model_size': 's',
                'pretrained': False,
                'freeze_backbone': False,
                'freeze_at': -1
            }
            backbone = YOLOv8Backbone(config).to(self.device)
        elif backbone_type == 'repvit':
            config = {
                'arch': 'm1',
                'img_size': 1024,
                'fuse': False,
                'freeze': False
            }
            backbone = RepViTBackbone(config).to(self.device)
        else:
            raise ValueError(f"不支持的backbone: {backbone_type}")
        
        return backbone
    
    def test_forward_pass(self, backbone, edge_head, feature_adapter):
        """测试前向传播"""
        print(f"🧪 测试前向传播...")
        
        backbone.eval()
        edge_head.eval()
        feature_adapter.eval()
        
        # 创建测试输入
        batch_size = 2
        test_image = torch.randn(batch_size, 3, 1024, 1024).to(self.device)
        test_teacher_features = torch.randn(batch_size, 256, 64, 64).to(self.device)
        test_edge_gt = torch.randint(0, 2, (batch_size, 256, 256)).float().to(self.device)
        
        with torch.no_grad():
            # Backbone前向传播
            backbone_features = backbone(test_image)
            print(f"   - Backbone输出: {len(backbone_features)} 个特征图")
            print(f"   - P3特征形状: {backbone_features[0].shape}")
            
            # Edge head前向传播
            edge_logits = edge_head(backbone_features[0], backbone_name='yolov8')
            print(f"   - Edge head输出形状: {edge_logits.shape}")
            
            # Feature adapter前向传播
            aligned_features = feature_adapter(backbone_features[0], test_teacher_features)
            print(f"   - 对齐特征形状: {aligned_features.shape}")
            
            # 损失计算测试
            edge_loss = EdgeDistillationLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)
            feature_loss = FeatureLoss({'loss_type': 'mse', 'alpha': 1.0}).to(self.device)
            
            edge_logits_resized = torch.nn.functional.interpolate(
                edge_logits, size=test_edge_gt.shape[1:], mode='bilinear', align_corners=False
            )
            
            edge_loss_val = edge_loss(edge_logits_resized, test_edge_gt)
            feature_loss_val = feature_loss(aligned_features, test_teacher_features)
            
            print(f"   - Edge损失: {edge_loss_val.item():.4f}")
            print(f"   - Feature损失: {feature_loss_val.item():.4f}")
        
        print(f"🎉 前向传播测试成功!")
        return True


def main():
    """主函数"""
    print("=== 智能权重加载测试 ===")
    
    checkpoint_path = "outputs/warmup_training/best_warmup_model.pth"
    loader = SmartWeightLoader()
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 权重文件不存在: {checkpoint_path}")
        return
    
    print(f"📁 使用权重文件: {checkpoint_path}")
    print(f"📊 文件大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")
    print()
    
    # 测试不同场景
    scenarios = [
        ("场景1: 预训练backbone加载", lambda: loader.load_for_pretrained_backbone(
            checkpoint_path, backbone_type='yolov8', load_edge_head=True, load_adapters=False
        )),
        ("场景2: 训练中断恢复", lambda: loader.load_for_training_resume(
            checkpoint_path, backbone_type='yolov8'
        )),
        ("场景3: 切换backbone", lambda: loader.load_for_backbone_switch(
            checkpoint_path, from_backbone='yolov8', to_backbone='repvit'
        ))
    ]
    
    for scenario_name, load_func in scenarios:
        print(f"\n{'='*60}")
        print(f"🧪 {scenario_name}")
        print(f"{'='*60}")
        
        try:
            result = load_func()
            
            if len(result) == 3:  # 场景1和3
                backbone, edge_head, feature_adapter = result
                loader.test_forward_pass(backbone, edge_head, feature_adapter)
            elif len(result) == 4:  # 场景2
                backbone, edge_head, feature_adapter, optimizer = result
                loader.test_forward_pass(backbone, edge_head, feature_adapter)
                print(f"✅ Optimizer状态恢复成功")
            
            print(f"🎉 {scenario_name} 测试通过!")
            
        except Exception as e:
            print(f"❌ {scenario_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    print(f"\n🎯 智能权重加载策略总结:")
    print(f"   - Backbone: 核心组件，预训练/训练中断必须加载")
    print(f"   - Edge Head: 核心通用组件，可以跨backbone复用")
    print(f"   - Adapters: 动态组件，换backbone需要重新创建")
    print(f"   - Optimizer: 仅训练中断需要加载")


if __name__ == "__main__":
    main()









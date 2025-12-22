#!/usr/bin/env python3
"""
权重加载测试脚本
测试从保存的checkpoint加载权重并验证功能
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


def test_checkpoint_loading(checkpoint_path, backbone_type='yolov8'):
    """测试checkpoint加载"""
    print(f"=== 测试权重加载: {checkpoint_path} ===")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 权重文件不存在: {checkpoint_path}")
        return False
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 成功加载checkpoint")
        print(f"   - 文件大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")
        
        # 检查checkpoint内容
        if 'backbone_state_dict' in checkpoint:
            print(f"   - 包含backbone权重: ✅")
        if 'edge_head_state_dict' in checkpoint:
            print(f"   - 包含edge_head权重: ✅")
        if 'feature_adapter_state_dict' in checkpoint:
            print(f"   - 包含feature_adapter权重: ✅")
        if 'optimizer_state_dict' in checkpoint:
            print(f"   - 包含optimizer状态: ✅")
        
        # 创建模型并加载权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - 使用设备: {device}")
        
        # 创建backbone
        if backbone_type == 'yolov8':
            backbone_config = {
                'model_size': 's',
                'pretrained': False,
                'freeze_backbone': False,
                'freeze_at': -1
            }
            backbone = YOLOv8Backbone(backbone_config).to(device)
        elif backbone_type == 'repvit':
            backbone_config = {
                'arch': 'm1',
                'img_size': 1024,
                'fuse': False,
                'freeze': False
            }
            backbone = RepViTBackbone(backbone_config).to(device)
        
        # 创建其他组件
        edge_head = UniversalEdgeHead(
            core_channels=64,
            output_channels=1,
            head_type='simple',
            init_p=0.05
        ).to(device)
        
        feature_adapter = SimpleAdapter().to(device)
        
        # 加载权重
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        
        # Edge head使用strict=False，因为动态通道对齐器可能不匹配
        missing_keys, unexpected_keys = edge_head.load_state_dict(
            checkpoint['edge_head_state_dict'], strict=False
        )
        if missing_keys:
            print(f"   - Edge head缺失键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"   - Edge head多余键: {len(unexpected_keys)} 个")
        
        # SimpleAdapter也使用strict=False，因为动态适配器可能不匹配
        missing_keys, unexpected_keys = feature_adapter.load_state_dict(
            checkpoint['feature_adapter_state_dict'], strict=False
        )
        if missing_keys:
            print(f"   - Feature adapter缺失键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"   - Feature adapter多余键: {len(unexpected_keys)} 个")
        
        print(f"✅ 成功加载所有模型权重")
        
        # 测试前向传播
        print(f"🧪 测试前向传播...")
        backbone.eval()
        edge_head.eval()
        feature_adapter.eval()
        
        # 创建测试输入
        batch_size = 2
        test_image = torch.randn(batch_size, 3, 1024, 1024).to(device)
        test_teacher_features = torch.randn(batch_size, 256, 64, 64).to(device)
        test_edge_gt = torch.randint(0, 2, (batch_size, 256, 256)).float().to(device)
        
        with torch.no_grad():
            # Backbone前向传播
            backbone_features = backbone(test_image)
            print(f"   - Backbone输出: {len(backbone_features)} 个特征图")
            print(f"   - P3特征形状: {backbone_features[0].shape}")
            
            # Edge head前向传播
            edge_logits = edge_head(backbone_features[0], backbone_name=backbone_type)
            print(f"   - Edge head输出形状: {edge_logits.shape}")
            
            # Feature adapter前向传播
            aligned_features = feature_adapter(backbone_features[0], test_teacher_features)
            print(f"   - 对齐特征形状: {aligned_features.shape}")
            
            # 损失计算测试
            edge_loss = EdgeDistillationLoss(bce_weight=0.5, dice_weight=0.5).to(device)
            feature_loss = FeatureLoss({'loss_type': 'mse', 'alpha': 1.0}).to(device)
            
            # 调整edge_logits尺寸以匹配GT
            edge_logits_resized = torch.nn.functional.interpolate(
                edge_logits, size=test_edge_gt.shape[1:], mode='bilinear', align_corners=False
            )
            
            edge_loss_val = edge_loss(edge_logits_resized, test_edge_gt)
            feature_loss_val = feature_loss(aligned_features, test_teacher_features)
            
            print(f"   - Edge损失: {edge_loss_val.item():.4f}")
            print(f"   - Feature损失: {feature_loss_val.item():.4f}")
        
        print(f"🎉 权重加载测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 权重加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=== 权重加载测试 ===")
    
    checkpoint_dir = "outputs/warmup_training"
    
    # 测试不同的checkpoint
    checkpoints_to_test = [
        ("best_warmup_model.pth", "最佳模型"),
        ("epoch_0_checkpoint.pth", "Epoch 0"),
        ("epoch_2_checkpoint.pth", "Epoch 2"),
        ("epoch_4_checkpoint.pth", "Epoch 4 (最终)")
    ]
    
    success_count = 0
    total_count = len(checkpoints_to_test)
    
    for checkpoint_file, description in checkpoints_to_test:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        print(f"\n📁 测试 {description}: {checkpoint_file}")
        
        if test_checkpoint_loading(checkpoint_path, backbone_type='yolov8'):
            success_count += 1
        
        print("-" * 60)
    
    print(f"\n📊 测试结果: {success_count}/{total_count} 个checkpoint加载成功")
    
    if success_count == total_count:
        print("🎉 所有权重加载测试通过!")
    else:
        print("⚠️  部分权重加载测试失败")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
测试backbone权重保存和加载功能
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.backbones.repvit_backbone import RepViTBackbone


def test_backbone_save_load():
    """测试backbone权重保存和加载"""
    print("=== 测试Backbone权重保存和加载 ===")
    
    # 测试YOLOv8
    print("\n1. 测试YOLOv8Backbone")
    yolov8_config = {
        'model_size': 's',
        'pretrained': False,
        'freeze_backbone': False,
        'freeze_at': -1
    }
    
    # 创建YOLOv8实例
    yolov8 = YOLOv8Backbone(yolov8_config)
    print(f"YOLOv8参数数量: {sum(p.numel() for p in yolov8.parameters()):,}")
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 1024, 1024)
    with torch.no_grad():
        features = yolov8(test_input)
    print(f"YOLOv8输出特征数量: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  特征{i}: {feat.shape}")
    
    # 保存权重
    save_path = "outputs/test_weights/yolov8_backbone.pth"
    yolov8.save_weights(save_path)
    
    # 创建新的YOLOv8实例并加载权重
    yolov8_new = YOLOv8Backbone(yolov8_config)
    yolov8_new.load_weights(save_path)
    
    # 验证权重是否一致
    with torch.no_grad():
        features_original = yolov8(test_input)
        features_loaded = yolov8_new(test_input)
    
    # 比较输出
    for i, (orig, loaded) in enumerate(zip(features_original, features_loaded)):
        diff = torch.abs(orig - loaded).max().item()
        print(f"  特征{i}最大差异: {diff:.6f}")
        assert diff < 1e-6, f"特征{i}差异过大: {diff}"
    
    print("✅ YOLOv8权重保存和加载测试通过！")
    
    # 测试RepViT
    print("\n2. 测试RepViTBackbone")
    repvit_config = {
        'model_size': 'm1',
        'pretrained': False,
        'freeze_backbone': False,
        'freeze_at': -1
    }
    
    # 创建RepViT实例
    repvit = RepViTBackbone(repvit_config)
    print(f"RepViT参数数量: {sum(p.numel() for p in repvit.parameters()):,}")
    
    # 测试前向传播
    with torch.no_grad():
        features = repvit(test_input)
    print(f"RepViT输出特征数量: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  特征{i}: {feat.shape}")
    
    # 保存权重
    save_path = "outputs/test_weights/repvit_backbone.pth"
    repvit.save_weights(save_path)
    
    # 创建新的RepViT实例并加载权重
    repvit_new = RepViTBackbone(repvit_config)
    repvit_new.load_weights(save_path)
    
    # 验证权重是否一致
    with torch.no_grad():
        features_original = repvit(test_input)
        features_loaded = repvit_new(test_input)
    
    # 比较输出
    for i, (orig, loaded) in enumerate(zip(features_original, features_loaded)):
        diff = torch.abs(orig - loaded).max().item()
        print(f"  特征{i}最大差异: {diff:.6f}")
        assert diff < 1e-6, f"特征{i}差异过大: {diff}"
    
    print("✅ RepViT权重保存和加载测试通过！")
    
    # 测试from_pretrained方法
    print("\n3. 测试from_pretrained方法")
    
    # 从YOLOv8权重创建新实例
    yolov8_from_pretrained = YOLOv8Backbone.from_pretrained("outputs/test_weights/yolov8_backbone.pth")
    
    # 验证输出
    with torch.no_grad():
        features_pretrained = yolov8_from_pretrained(test_input)
        features_yolov8_orig = yolov8(test_input)  # 重新获取原始特征
    
    for i, (orig, pretrained) in enumerate(zip(features_yolov8_orig, features_pretrained)):
        diff = torch.abs(orig - pretrained).max().item()
        print(f"  特征{i}最大差异: {diff:.6f}")
        assert diff < 1e-6, f"特征{i}差异过大: {diff}"
    
    print("✅ from_pretrained方法测试通过！")
    
    print("\n🎉 所有backbone权重保存和加载测试通过！")


def test_model_info():
    """测试模型信息功能"""
    print("\n=== 测试模型信息功能 ===")
    
    # YOLOv8信息
    yolov8_config = {'model_size': 's', 'pretrained': False, 'freeze_backbone': False, 'freeze_at': -1}
    yolov8 = YOLOv8Backbone(yolov8_config)
    info = yolov8.get_model_info()
    print("YOLOv8模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # RepViT信息
    repvit_config = {'model_size': 'm1', 'pretrained': False, 'freeze_backbone': False, 'freeze_at': -1}
    repvit = RepViTBackbone(repvit_config)
    info = repvit.get_model_info()
    print("\nRepViT模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_backbone_save_load()
    test_model_info()

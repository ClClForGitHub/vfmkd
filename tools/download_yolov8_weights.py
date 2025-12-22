"""
自动下载YOLOv8官方权重到weights目录
"""

import os
import sys
from pathlib import Path

import torch
import torch.hub

from vfmkd.utils.path_setup import ensure_project_paths
ensure_project_paths()


def download_yolov8_weights():
    """下载YOLOv8s权重到weights目录"""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    weight_path = weights_dir / "yolov8s.pt"
    
    if weight_path.exists():
        print(f"权重已存在: {weight_path}")
        return str(weight_path)
    
    print("正在下载YOLOv8s权重...")
    try:
        # 使用torch.hub下载官方权重
        model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)
        torch.save(model.state_dict(), weight_path)
        print(f"权重下载完成: {weight_path}")
        return str(weight_path)
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试备用方法...")
        
        # 备用方法：直接下载权重文件
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        try:
            urllib.request.urlretrieve(url, weight_path)
            print(f"权重下载完成: {weight_path}")
            return str(weight_path)
        except Exception as e2:
            print(f"备用下载也失败: {e2}")
            return None


def test_weight_loading(weight_path: str):
    """测试权重是否能正确加载到我们的backbone"""
    if not weight_path or not os.path.exists(weight_path):
        print("权重文件不存在，无法测试")
        return False
    
    try:
        from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
        
        # 创建backbone并加载权重
        backbone = YOLOv8Backbone({
            'model_size': 's',
            'pretrained': False,
            'external_weight_path': weight_path,
        })
        
        # 测试前向传播
        x = torch.randn(1, 3, 1024, 1024)
        feats = backbone(x)
        
        print(f"Backbone测试成功:")
        print(f"  输入形状: {x.shape}")
        print(f"  输出特征数: {len(feats)}")
        for i, feat in enumerate(feats):
            print(f"  feat_{i}: {feat.shape}")
        
        return True
    except Exception as e:
        print(f"Backbone测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=== YOLOv8权重下载与测试 ===")
    
    # 下载权重
    weight_path = download_yolov8_weights()
    
    if weight_path:
        # 测试加载
        print("\n=== 测试权重加载 ===")
        test_weight_loading(weight_path)
    else:
        print("权重下载失败，请手动下载")
        sys.exit(1)

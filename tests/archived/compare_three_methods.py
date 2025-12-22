#!/usr/bin/env python3
"""
对比MSE、FGD、FGD+Edge三种方法的效果
评估维度：
1. 特征对齐质量（与教师特征的相似度）
2. 边缘预测质量
3. 训练loss对比
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.universal_edge_head import UniversalEdgeHead
from vfmkd.distillation.adapters import SimpleAdapter
from torch.utils.data import Dataset, DataLoader
import cv2


class NPZDataset(Dataset):
    """加载NPZ特征和真实图像"""
    def __init__(self, npz_dir: str, images_dir: str, input_size: int = 1024):
        self.npz_dir = Path(npz_dir)
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        
        # 收集所有NPZ文件
        self.npz_files = sorted(list(self.npz_dir.glob("sa_*_features.npz")))
        print(f"Found {len(self.npz_files)} NPZ files")
    
    def __len__(self):
        return len(self.npz_files)
    
    def _load_real_image(self, image_id: str) -> np.ndarray:
        """加载真实图像"""
        possible_names = [
            f"sa_{image_id}.jpg",
            f"{image_id}.jpg",
            f"sa_{image_id}.png",
            f"{image_id}.png",
        ]
        
        for name in possible_names:
            img_path = self.images_dir / name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.input_size, self.input_size))
                    img = img.astype(np.float32) / 255.0
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    return img
        
        raise FileNotFoundError(f"Image not found for ID: {image_id}")
    
    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        
        # 提取image_id
        image_id = npz_file.stem.replace("_features", "").replace("sa_", "")
        
        # 加载教师特征
        if 'P4_S16' in data:
            teacher_features = torch.from_numpy(data['P4_S16'])
        elif 'IMAGE_EMB_S16' in data:
            teacher_features = torch.from_numpy(data['IMAGE_EMB_S16'])
        else:
            raise KeyError(f"No teacher features found in {npz_file}")
        
        # 加载边缘GT
        edge_256x256 = torch.from_numpy(data['edge_256x256']).float()
        
        # 加载真实图像
        image = self._load_real_image(image_id)
        
        return {
            "image": image,
            "teacher_features": teacher_features,
            "edge_256x256": edge_256x256,
            "image_id": image_id
        }


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载训练好的模型"""
    # 创建模型
    backbone = YOLOv8Backbone({
        "model_size": "s",
        "pretrained": False,
        "freeze_backbone": False,
    }).to(device)
    
    edge_head = UniversalEdgeHead(
        core_channels=64,
        head_type='simple'
    ).to(device)
    
    feature_adapter = SimpleAdapter(
        student_channels=256,
        teacher_channels=256
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone'], strict=True)
    edge_head.load_state_dict(checkpoint['edge_head'], strict=False)
    feature_adapter.load_state_dict(checkpoint['feature_adapter'], strict=True)
    
    backbone.eval()
    edge_head.eval()
    feature_adapter.eval()
    
    return backbone, edge_head, feature_adapter


def compute_feature_metrics(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> Dict[str, float]:
    """计算特征对齐指标"""
    # MSE
    mse = F.mse_loss(student_feat, teacher_feat).item()
    
    # MAE
    mae = F.l1_loss(student_feat, teacher_feat).item()
    
    # 余弦相似度
    student_flat = student_feat.flatten(1)
    teacher_flat = teacher_feat.flatten(1)
    cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1).mean().item()
    
    # 相对误差
    relative_error = (torch.abs(student_feat - teacher_feat) / (torch.abs(teacher_feat) + 1e-8)).mean().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "relative_error": relative_error
    }


def compute_edge_metrics(pred_edge: torch.Tensor, gt_edge: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """计算边缘预测指标"""
    # 二值化
    pred_binary = (torch.sigmoid(pred_edge) > threshold).float()
    gt_binary = (gt_edge > 0.5).float()
    
    # IoU
    intersection = (pred_binary * gt_binary).sum()
    union = (pred_binary + gt_binary).clamp(0, 1).sum()
    iou = (intersection / (union + 1e-8)).item()
    
    # Dice
    dice = (2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)).item()
    
    # 精确率和召回率
    tp = (pred_binary * gt_binary).sum().item()
    fp = (pred_binary * (1 - gt_binary)).sum().item()
    fn = ((1 - pred_binary) * gt_binary).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


@torch.no_grad()
def evaluate_model(model_name: str, checkpoint_path: str, test_loader: DataLoader, device: str = 'cuda'):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    backbone, edge_head, feature_adapter = load_model(checkpoint_path, device)
    
    # 累积指标
    feature_metrics_list = []
    edge_metrics_list = []
    
    for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
        images = batch["image"].to(device)
        teacher_features = batch["teacher_features"].to(device)
        edge_gt = batch["edge_256x256"].to(device)
        
        # 前向传播
        features = backbone(images)
        s16_features = features[1]  # (B, 256, 64, 64)
        
        # 特征对齐
        aligned_features = feature_adapter(s16_features, teacher_features)
        
        # 边缘预测
        edge_logits = edge_head(s16_features, backbone_name='yolov8')
        if edge_logits.shape[2:] != edge_gt.shape[1:]:
            edge_logits = F.interpolate(edge_logits, size=edge_gt.shape[1:], mode='bilinear', align_corners=False)
        
        # 计算指标
        feature_metrics = compute_feature_metrics(aligned_features, teacher_features)
        edge_metrics = compute_edge_metrics(edge_logits, edge_gt.unsqueeze(1))
        
        feature_metrics_list.append(feature_metrics)
        edge_metrics_list.append(edge_metrics)
    
    # 平均指标
    avg_feature_metrics = {
        k: np.mean([m[k] for m in feature_metrics_list]) 
        for k in feature_metrics_list[0].keys()
    }
    avg_edge_metrics = {
        k: np.mean([m[k] for m in edge_metrics_list]) 
        for k in edge_metrics_list[0].keys()
    }
    
    return {
        "feature": avg_feature_metrics,
        "edge": avg_edge_metrics
    }


def print_comparison_table(results: Dict[str, Dict]):
    """打印对比表格"""
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # 特征对齐指标
    print(f"\n{'Feature Alignment Metrics (Lower is Better for MSE/MAE/Error, Higher for Cosine)'}")
    print(f"{'-'*80}")
    print(f"{'Method':<20} {'MSE':<12} {'MAE':<12} {'Cosine Sim':<12} {'Rel Error':<12}")
    print(f"{'-'*80}")
    for method, metrics in results.items():
        feat = metrics['feature']
        print(f"{method:<20} {feat['mse']:<12.6f} {feat['mae']:<12.6f} {feat['cosine_similarity']:<12.6f} {feat['relative_error']:<12.6f}")
    
    # 边缘预测指标
    print(f"\n{'Edge Prediction Metrics (Higher is Better)'}")
    print(f"{'-'*80}")
    print(f"{'Method':<20} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*80}")
    for method, metrics in results.items():
        edge = metrics['edge']
        print(f"{method:<20} {edge['iou']:<10.4f} {edge['dice']:<10.4f} {edge['precision']:<10.4f} {edge['recall']:<10.4f} {edge['f1']:<10.4f}")
    
    print(f"{'='*80}\n")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 路径配置
    npz_dir = "VFMKD/outputs/features_v1_300"
    images_dir = "datasets/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0"
    
    # 模型checkpoint路径
    models = {
        "MSE": "outputs/distill_single_test/mse_best.pth",
        "FGD": "outputs/distill_single_test/fgd_best.pth",
        "FGD+Edge": "outputs/distill_single_test/fgd_edge_best.pth"
    }
    
    # 创建测试数据集（使用固定的测试集）
    dataset = NPZDataset(npz_dir, images_dir)
    
    # 使用相同的随机种子划分测试集
    torch.manual_seed(42)
    total_size = len(dataset)
    test_size = int(0.1 * total_size)
    train_size = total_size - test_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Test set size: {len(test_dataset)}")
    
    # 评估所有模型
    results = {}
    for model_name, checkpoint_path in models.items():
        if not Path(checkpoint_path).exists():
            print(f"[WARN] Checkpoint not found: {checkpoint_path}, skipping {model_name}")
            continue
        
        try:
            results[model_name] = evaluate_model(model_name, checkpoint_path, test_loader, device)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            continue
    
    # 打印对比表格
    if results:
        print_comparison_table(results)
        
        # 保存结果到JSON
        output_file = "VFMKD/outputs/testFGDFSD/comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        print("[ERROR] No results to display")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
用MSE损失对单张图片进行过拟合训练
目标：让RepViT+Adapter学习SAM2特征的数值分布（而不只是方向）
"""
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine']:
    if _mod not in sys.modules:
        sys.modules[_mod] = _types.ModuleType(_mod)

from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter


class SingleImageDataset(Dataset):
    """只包含单张图片的数据集"""
    def __init__(self, image_path: str, npz_path: str, transform=None):
        self.image_path = image_path
        self.npz_path = npz_path
        self.transform = transform
        
        # 加载图片
        self.image = Image.open(image_path).convert('RGB')
        
        # 加载教师特征
        npz = np.load(npz_path)
        if 'IMAGE_EMB_S16' in npz:
            self.teacher_feat = torch.from_numpy(npz['IMAGE_EMB_S16'])
        elif 'image_embedding' in npz:
            self.teacher_feat = torch.from_numpy(npz['image_embedding'])
        else:
            raise KeyError(f"NPZ文件中找不到特征键，可用的键: {list(npz.keys())}")
        
        print(f"[INFO] 教师特征 shape: {self.teacher_feat.shape}")
        print(f"[INFO] 教师特征统计: mean={self.teacher_feat.mean():.4f}, std={self.teacher_feat.std():.4f}")
        print(f"[INFO] 教师特征范围: [{self.teacher_feat.min():.4f}, {self.teacher_feat.max():.4f}]")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        img = self.image
        if self.transform:
            img = self.transform(img)
        return img, self.teacher_feat, self.image_path


def forward_backbone(backbone, x):
    """获取backbone的最后一层特征"""
    feats = backbone(x)
    if isinstance(feats, list):
        return feats[-1]
    return feats


def main():
    parser = argparse.ArgumentParser(description='用MSE损失对单张图片过拟合')
    parser.add_argument('--image', type=str, 
                        default='datasets/coco128/images/train2017/000000000009.jpg',
                        help='训练图片路径')
    parser.add_argument('--npz', type=str,
                        default='datasets/coco128/SAM_Cache/000000000009_sam2_features.npz',
                        help='教师特征NPZ路径')
    parser.add_argument('--iterations', type=int, default=1000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_every', type=int, default=10, help='每N次打印一次')
    parser.add_argument('--save_weights', type=str, 
                        default='outputs/overfit_mse_weights.pth',
                        help='保存训练后的权重到指定路径')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("[INFO] 使用设备:", device)
    print("[INFO] 图片:", args.image)
    print("[INFO] NPZ:", args.npz)
    print("[INFO] 迭代次数:", args.iterations)
    print("[INFO] 学习率:", args.lr)
    print("[INFO] 损失函数: MSE (Mean Squared Error)")
    print("="*80)
    
    # 数据变换
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    dataset = SingleImageDataset(args.image, args.npz, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 初始化模型
    print("[INFO] 初始化 RepViT backbone...")
    config = {
        'arch': 'm1',
        'img_size': 1024,
        'fuse': False,
        'freeze': False,
        'load_from': None
    }
    backbone = RepViTBackbone(config).to(device)
    
    print("[INFO] 初始化 Adapter...")
    adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256).to(device)
    
    # MSE损失函数
    mse_loss = nn.MSELoss()
    
    # 优化器
    params = list(backbone.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # 训练模式
    backbone.train()
    adapter.train()
    
    # 获取单张图片和教师特征
    x, teacher, img_path = next(iter(dataloader))
    x = x.to(device)
    teacher = teacher.to(device)
    
    print(f"[INFO] 输入 shape: {x.shape}")
    print(f"[INFO] 教师特征 shape: {teacher.shape}")
    print("="*80)
    print("[INFO] 开始训练...")
    print("="*80)
    
    # 训练循环
    pbar = tqdm(range(1, args.iterations + 1), desc="Training")
    for iteration in pbar:
        optimizer.zero_grad()
        
        # 前向传播
        feat = forward_backbone(backbone, x)
        student = adapter(feat)
        
        # 确保尺寸匹配
        if student.shape != teacher.shape:
            student = F.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
        
        # MSE损失
        loss = mse_loss(student, teacher)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 每N次打印统计信息
        if iteration % args.print_every == 0:
            with torch.no_grad():
                s_mean = student.mean().item()
                s_std = student.std().item()
                s_min = student.min().item()
                s_max = student.max().item()
                t_mean = teacher.mean().item()
                t_std = teacher.std().item()
                t_min = teacher.min().item()
                t_max = teacher.max().item()
                
                # 计算额外指标
                abs_diff = torch.abs(student - teacher)
                cosine_sim = F.cosine_similarity(
                    student.flatten(),
                    teacher.flatten(),
                    dim=0
                ).item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "s_mean": f"{s_mean:.3f}",
                    "t_mean": f"{t_mean:.3f}",
                    "cosine": f"{cosine_sim:.4f}",
                    "max_diff": f"{abs_diff.max().item():.3f}"
                })
    
    # 最终评估
    print("\n" + "="*80)
    print("[INFO] 训练完成！最终统计:")
    print("="*80)
    
    backbone.eval()
    adapter.eval()
    
    with torch.no_grad():
        feat = forward_backbone(backbone, x)
        student = adapter(feat)
        if student.shape != teacher.shape:
            student = F.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
        
        final_loss = mse_loss(student, teacher).item()
        abs_diff = torch.abs(student - teacher)
        cosine_sim = F.cosine_similarity(student.flatten(), teacher.flatten(), dim=0).item()
        
        print(f"最终MSE损失: {final_loss:.6f}")
        print(f"余弦相似度: {cosine_sim:.6f}")
        print(f"\n学生特征统计:")
        print(f"  Mean: {student.mean():.4f}, Std: {student.std():.4f}")
        print(f"  Range: [{student.min():.4f}, {student.max():.4f}]")
        print(f"\n教师特征统计:")
        print(f"  Mean: {teacher.mean():.4f}, Std: {teacher.std():.4f}")
        print(f"  Range: [{teacher.min():.4f}, {teacher.max():.4f}]")
        print(f"\n差异统计:")
        print(f"  绝对差异: mean={abs_diff.mean():.4f}, max={abs_diff.max():.4f}")
        print(f"  相对误差: {(abs_diff.mean() / teacher.abs().mean() * 100):.2f}%")
    
    # 保存权重
    if args.save_weights:
        print(f"\n[INFO] 保存权重到: {args.save_weights}")
        save_dict = {
            'backbone': backbone.state_dict(),
            'adapter': adapter.state_dict(),
            'final_loss': final_loss,
            'cosine_similarity': cosine_sim,
        }
        torch.save(save_dict, args.save_weights)
        print("[INFO] 权重保存成功!")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == '__main__':
    main()


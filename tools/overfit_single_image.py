#!/usr/bin/env python3
"""
单张图片过拟合测试 (Sanity Check)
验证模型是否有能力拟合单张图片的特征
"""
import argparse
import sys
from pathlib import Path

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

from vfmkd.models.backbones.repvit_backbone import RepViT, RepViTBackbone
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
        data = np.load(npz_path)
        # 尝试不同的键名
        if 'IMAGE_EMB_S16' in data:
            self.teacher_feat = torch.from_numpy(data['IMAGE_EMB_S16']).float()
        elif 'image_embedding' in data:
            self.teacher_feat = torch.from_numpy(data['image_embedding']).float()
        else:
            raise KeyError(f"在NPZ文件中找不到特征键，可用键: {list(data.keys())}")
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        img = self.image
        if self.transform:
            img = self.transform(img)
        return img, self.teacher_feat, self.image_path


def cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """余弦相似度损失"""
    s = F.normalize(student.flatten(1), dim=1)
    t = F.normalize(teacher.flatten(1), dim=1)
    return 1.0 - (s * t).sum(dim=1).mean()


def forward_backbone(backbone, x):
    """根据backbone类型调用前向传播"""
    if hasattr(backbone, 'extract_features'):
        return backbone.extract_features(x)
    elif hasattr(backbone, 'forward_features'):
        return backbone.forward_features(x)
    else:
        return backbone(x)


def main():
    parser = argparse.ArgumentParser(description='单张图片过拟合测试')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--npz', type=str, required=True, help='对应的NPZ特征文件路径')
    parser.add_argument('--backbone', type=str, default='repvit', choices=['repvit'])
    parser.add_argument('--iterations', type=int, default=500, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--normalize', action='store_true', help='是否在适配器后添加归一化')
    parser.add_argument('--print_every', type=int, default=10, help='每N次打印一次')
    parser.add_argument('--save_weights', type=str, default='', help='保存训练后的权重到指定路径')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 图片: {args.image}")
    print(f"[INFO] NPZ: {args.npz}")
    print(f"[INFO] 迭代次数: {args.iterations}")
    print(f"[INFO] 学习率: {args.lr}")
    print(f"[INFO] 归一化: {args.normalize}")
    print("=" * 80)

    # 数据变换
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
    ])

    # 创建数据集和数据加载器
    dataset = SingleImageDataset(args.image, args.npz, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化模型
    if args.backbone == 'repvit':
        print("[INFO] 初始化 RepViT backbone...")
        config = {
            'arch': 'm1',
            'img_size': 1024,
            'fuse': False,
            'freeze': False,
            'load_from': None
        }
        backbone = RepViTBackbone(config)
        
        print("[INFO] 初始化 RepViTAlignAdapter...")
        adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256)
    else:
        raise ValueError(f"不支持的backbone: {args.backbone}")

    backbone = backbone.to(device)
    adapter = adapter.to(device)

    # 如果需要归一化，添加额外的归一化层
    if args.normalize:
        print("[INFO] 添加额外的 L2 归一化层...")
        normalizer = nn.Sequential(
            nn.LayerNorm([256, 64, 64]),
            nn.Identity()  # 占位，可以改为其他归一化
        ).to(device)
    else:
        normalizer = None

    # 优化器
    params = list(backbone.parameters()) + list(adapter.parameters())
    if normalizer:
        params += list(normalizer.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # 训练模式
    backbone.train()
    adapter.train()
    if normalizer:
        normalizer.train()

    # 获取单张图片和教师特征
    x, teacher, img_path = next(iter(dataloader))
    x = x.to(device)
    teacher = teacher.to(device)
    
    print(f"[INFO] 图片 shape: {x.shape}")
    print(f"[INFO] 教师特征 shape: {teacher.shape}")
    print(f"[INFO] 教师特征统计: mean={teacher.mean().item():.4f}, std={teacher.std().item():.4f}, "
          f"min={teacher.min().item():.4f}, max={teacher.max().item():.4f}")
    print("=" * 80)

    # 过拟合循环
    print(f"[INFO] 开始过拟合训练...")
    losses = []
    
    pbar = tqdm(range(args.iterations), desc="Overfitting")
    for iteration in pbar:
        optimizer.zero_grad()
        
        # 前向传播
        feats = forward_backbone(backbone, x)
        student = adapter(feats)
        
        # 如果有归一化器
        if normalizer:
            student = normalizer(student)
        
        # 确保尺寸匹配
        if student.shape != teacher.shape:
            student = F.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
        
        # 计算损失
        loss = cosine_loss(student, teacher)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录损失
        loss_val = loss.item()
        losses.append(loss_val)
        
        # 定期打印
        if (iteration + 1) % args.print_every == 0 or iteration == 0:
            with torch.no_grad():
                s_mean = student.mean().item()
                s_std = student.std().item()
                s_min = student.min().item()
                s_max = student.max().item()
                
            pbar.set_postfix({
                'loss': f'{loss_val:.6f}',
                's_mean': f'{s_mean:.3f}',
                's_std': f'{s_std:.3f}',
                's_range': f'[{s_min:.2f},{s_max:.2f}]'
            })
            
            print(f"\n[Iter {iteration+1:4d}] Loss: {loss_val:.6f} | "
                  f"Student: mean={s_mean:.4f}, std={s_std:.4f}, range=[{s_min:.2f}, {s_max:.2f}]")

    print("=" * 80)
    print(f"[INFO] 过拟合完成!")
    print(f"[INFO] 初始损失: {losses[0]:.6f}")
    print(f"[INFO] 最终损失: {losses[-1]:.6f}")
    print(f"[INFO] 损失下降: {losses[0] - losses[-1]:.6f} ({(1 - losses[-1]/losses[0])*100:.2f}%)")
    
    # 最终特征统计
    with torch.no_grad():
        feats = forward_backbone(backbone, x)
        student = adapter(feats)
        if normalizer:
            student = normalizer(student)
        if student.shape != teacher.shape:
            student = F.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
        
        print("\n[INFO] 最终特征统计:")
        print(f"  学生: mean={student.mean().item():.4f}, std={student.std().item():.4f}, "
              f"min={student.min().item():.4f}, max={student.max().item():.4f}")
        print(f"  教师: mean={teacher.mean().item():.4f}, std={teacher.std().item():.4f}, "
              f"min={teacher.min().item():.4f}, max={teacher.max().item():.4f}")
    
    # 保存权重
    if args.save_weights:
        print(f"\n[INFO] 保存权重到: {args.save_weights}")
        save_dict = {
            'backbone': backbone.state_dict(),
            'adapter': adapter.state_dict(),
        }
        if normalizer:
            save_dict['normalizer'] = normalizer.state_dict()
        torch.save(save_dict, args.save_weights)
        print("[INFO] 权重保存成功!")


if __name__ == '__main__':
    main()


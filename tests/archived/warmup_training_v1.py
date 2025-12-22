#!/usr/bin/env python3
"""
第一版WARM-UP训练脚本
整合backbone+边缘头+MSE损失
支持可配置的backbone切换和损失权重设置
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.models.heads.edge_head import UniversalEdgeHead
from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.distillation.losses.edge_loss import EdgeDistillationLoss
from vfmkd.distillation.losses.feature_loss import FeatureLoss
from vfmkd.distillation.adapters import SimpleAdapter


class WarmUpDataset(Dataset):
    """WARM-UP训练数据集"""
    
    def __init__(self, features_dir, images_dir=None, max_images=None, input_size=1024):
        """
        Args:
            features_dir: 特征NPZ文件目录
            images_dir: 图像目录 (可选，如果不提供则使用随机图像)
            max_images: 最大图像数量
            input_size: 输入图像尺寸
        """
        self.features_dir = Path(features_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.input_size = input_size
        
        # 获取所有NPZ文件
        npz_files = list(self.features_dir.glob("*_features.npz"))
        if max_images:
            npz_files = npz_files[:max_images]
        
        # 过滤有效的NPZ文件
        self.valid_files = []
        for npz_file in npz_files:
            try:
                # 检查NPZ文件内容
                data = np.load(npz_file)
                required_keys = ['IMAGE_EMB_S16', 'edge_original', 'edge_256x256']
                if all(key in data for key in required_keys):
                    self.valid_files.append(npz_file)
                else:
                    print(f"⚠️  跳过 {npz_file.stem}: 缺少必要的数据")
            except Exception as e:
                print(f"⚠️  跳过 {npz_file.stem}: {e}")
        
        print(f"WARM-UP数据集: {len(self.valid_files)} 个有效样本")
        if self.images_dir:
            print(f"图像目录: {self.images_dir}")
        else:
            print("⚠️  未提供图像目录，将使用随机图像")
    
    def __len__(self):
        return len(self.valid_files)
    
    def _load_real_image(self, image_id):
        """加载真实图像"""
        if self.images_dir is None:
            # 如果没有提供图像目录，使用随机图像
            print(f"⚠️  使用随机图像 for {image_id}")
            return torch.randn(3, self.input_size, self.input_size)
        
        # 查找对应的图像文件
        image_file = self.images_dir / f"{image_id}.jpg"
        if not image_file.exists():
            print(f"⚠️  找不到图像文件 {image_file}，使用随机图像")
            return torch.randn(3, self.input_size, self.input_size)
        
        try:
            import cv2
            # 加载图像
            image = cv2.imread(str(image_file))
            if image is None:
                raise ValueError(f"无法加载图像: {image_file}")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # YOLO学生模型预处理：仅Resize + /255（无ImageNet标准化）
            # 调整尺寸到1024×1024
            image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
            
            # 转换为tensor并归一化到[0,1]
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            
            return image_tensor
            
        except Exception as e:
            print(f"⚠️  加载图像失败 {image_file}: {e}，使用随机图像")
            return torch.randn(3, self.input_size, self.input_size)
    
    def __getitem__(self, idx):
        npz_file = self.valid_files[idx]
        image_id = npz_file.stem.replace('_features', '')
        
        # 加载NPZ数据
        data = np.load(npz_file)
        
        # 提取数据
        teacher_features = torch.from_numpy(data['IMAGE_EMB_S16']).float()  # [1, 256, 64, 64]
        edge_original = torch.from_numpy(data['edge_original']).float()  # [H, W] 原图尺寸
        edge_256x256 = torch.from_numpy(data['edge_256x256']).float()  # [256, 256]
        
        # 加载真实图像
        real_image = self._load_real_image(image_id)
        
        return {
            'image': real_image,
            'teacher_features': teacher_features.squeeze(0),  # [256, 64, 64]
            'edge_256x256': edge_256x256,  # 只使用256x256边缘图，原图尺寸不同无法batch
            'image_id': image_id
        }


class WarmUpTrainer:
    """WARM-UP训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建backbone
        self.backbone = self._create_backbone()
        
        # 创建边缘头
        self.edge_head = UniversalEdgeHead(
            core_channels=config.get('core_channels', 64),
            output_channels=1,
            head_type=config.get('head_type', 'simple'),
            init_p=0.05
        ).to(self.device)
        
        # 创建特征对齐器
        self.feature_adapter = SimpleAdapter().to(self.device)
        
        # 创建损失函数
        self.edge_loss = EdgeDistillationLoss(
            bce_weight=config.get('bce_weight', 0.5),
            dice_weight=config.get('dice_weight', 0.5)
        ).to(self.device)
        
        self.feature_loss = FeatureLoss({
            'loss_type': 'mse',
            'alpha': config.get('mse_weight', 1.0)
        }).to(self.device)
        
        # 创建优化器
        all_params = (list(self.backbone.parameters()) + 
                     list(self.edge_head.parameters()) + 
                     list(self.feature_adapter.parameters()))
        self.optimizer = optim.Adam(
            all_params,
            lr=config.get('learning_rate', 0.001)
        )
        
        print(f"Backbone参数: {sum(p.numel() for p in self.backbone.parameters()):,}")
        print(f"边缘头参数: {sum(p.numel() for p in self.edge_head.parameters()):,}")
        print(f"特征对齐器参数: {sum(p.numel() for p in self.feature_adapter.parameters()):,}")
        print(f"总参数: {sum(p.numel() for p in all_params):,}")
    
    def _create_backbone(self):
        """创建backbone"""
        backbone_type = self.config.get('backbone', 'yolov8')
        
        if backbone_type == 'yolov8':
            backbone_config = {
                'model_size': 's',
                'pretrained': self.config.get('pretrained', False),
                'freeze_backbone': self.config.get('freeze_backbone', False),
                'freeze_at': self.config.get('freeze_at', -1),
                'checkpoint_path': self.config.get('checkpoint_path', None)
            }
            backbone = YOLOv8Backbone(backbone_config)
        elif backbone_type == 'repvit':
            backbone_config = {
                'arch': 'm1',
                'img_size': 1024,
                'fuse': False,
                'freeze': self.config.get('freeze_backbone', False),
                'load_from': self.config.get('checkpoint_path', None)
            }
            backbone = RepViTBackbone(backbone_config)
        else:
            raise ValueError(f"不支持的backbone: {backbone_type}")
        
        backbone = backbone.to(self.device)
        backbone.train()
        
        print(f"创建backbone: {backbone_type}")
        
        # 加载预训练权重
        checkpoint_path = self.config.get('checkpoint_path', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"加载预训练权重: {checkpoint_path}")
            if backbone_type == 'yolov8':
                backbone.load_pretrained(checkpoint_path)
            elif backbone_type == 'repvit':
                # RepViT的权重加载在初始化时已经处理
                pass
        elif self.config.get('pretrained', False):
            print("⚠️  配置了pretrained=True但没有提供checkpoint_path")
        
        return backbone
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.backbone.train()
        self.edge_head.train()
        self.feature_adapter.train()
        
        total_loss = 0
        total_mse_loss = 0
        total_edge_loss = 0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            teacher_features = batch['teacher_features'].to(self.device)  # [B, 256, 64, 64]
            edge_256x256 = batch['edge_256x256'].to(self.device)  # [B, 256, 256]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 1. Backbone特征提取
            features = self.backbone(images)
            p3_features = features[0]  # [B, C, H, W] - P3特征
            
            # 2. 特征对齐和MSE损失
            aligned_features = self.feature_adapter(p3_features, teacher_features)  # [B, 256, 64, 64]
            mse_loss = self.feature_loss(aligned_features, teacher_features)
            
            # 3. 边缘预测和边缘损失
            edge_logits = self.edge_head(p3_features, backbone_name=self.config.get('backbone', 'yolov8'))
            
            # 将边缘预测上采样到256×256
            if edge_logits.shape[2:] != edge_256x256.shape[1:]:
                edge_logits = torch.nn.functional.interpolate(
                    edge_logits, 
                    size=edge_256x256.shape[1:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            edge_loss_dict = self.edge_loss(
                edge_logits, 
                edge_256x256, 
                return_components=True
            )
            edge_loss_val = edge_loss_dict['total_loss']
            
            # 4. 总损失
            total_loss_val = (self.config.get('mse_weight', 1.0) * mse_loss + 
                            self.config.get('edge_weight', 1.0) * edge_loss_val)
            
            # 反向传播
            total_loss_val.backward()
            self.optimizer.step()
            
            # 累计损失
            total_loss += total_loss_val.item()
            total_mse_loss += mse_loss.item()
            total_edge_loss += edge_loss_val.item()
            
            # 更新进度条
            pbar.set_postfix({
                'total': f"{total_loss_val.item():.4f}",
                'mse': f"{mse_loss.item():.4f}",
                'edge': f"{edge_loss_val.item():.4f}",
                'avg_total': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_total_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        
        # 保存可视化
        self.save_visualization(epoch, images, edge_logits, edge_256x256, aligned_features, teacher_features)
        
        return avg_total_loss, avg_mse_loss, avg_edge_loss
    
    def save_visualization(self, epoch, images, edge_logits, edge_targets, aligned_features, teacher_features):
        """保存训练可视化"""
        vis_dir = Path("outputs/warmup_training/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 取第一个样本
        edge_pred = torch.sigmoid(edge_logits[0, 0]).detach().cpu().numpy()
        edge_target = edge_targets[0].detach().cpu().numpy()
        
        # 保存边缘图
        edge_pred_img = (edge_pred * 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"epoch_{epoch:02d}_edge_pred.png"), edge_pred_img)
        
        edge_target_img = (edge_target * 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"epoch_{epoch:02d}_edge_target.png"), edge_target_img)
        
        print(f"Epoch {epoch}: 可视化已保存到 {vis_dir}")
    
    def validate(self, dataloader):
        """验证"""
        self.backbone.eval()
        self.edge_head.eval()
        self.feature_adapter.eval()
        
        total_loss = 0
        total_mse_loss = 0
        total_edge_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                teacher_features = batch['teacher_features'].to(self.device)
                edge_256x256 = batch['edge_256x256'].to(self.device)
                
                # 前向传播
                features = self.backbone(images)
                p3_features = features[0]
                
                # MSE损失
                aligned_features = self.feature_adapter(p3_features, teacher_features)
                mse_loss = self.feature_loss(aligned_features, teacher_features)
                
                # 边缘损失
                edge_logits = self.edge_head(p3_features, backbone_name=self.config.get('backbone', 'yolov8'))
                
                if edge_logits.shape[2:] != edge_256x256.shape[1:]:
                    edge_logits = torch.nn.functional.interpolate(
                        edge_logits, 
                        size=edge_256x256.shape[1:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                edge_loss_dict = self.edge_loss(
                    edge_logits, 
                    edge_256x256, 
                    return_components=True
                )
                edge_loss_val = edge_loss_dict['total_loss']
                
                # 总损失
                total_loss_val = (self.config.get('mse_weight', 1.0) * mse_loss + 
                                self.config.get('edge_weight', 1.0) * edge_loss_val)
                
                total_loss += total_loss_val.item()
                total_mse_loss += mse_loss.item()
                total_edge_loss += edge_loss_val.item()
        
        avg_total_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches
        
        return avg_total_loss, avg_mse_loss, avg_edge_loss
    
    def train(self, train_loader, val_loader, epochs):
        """训练WARM-UP模型"""
        print(f"开始WARM-UP训练，共 {epochs} 个epoch")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            train_total, train_mse, train_edge = self.train_epoch(train_loader, epoch + 1)
            
            # 验证
            val_total, val_mse, val_edge = self.validate(val_loader)
            
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train - Total: {train_total:.4f}, MSE: {train_mse:.4f}, Edge: {train_edge:.4f}")
            print(f"  Val   - Total: {val_total:.4f}, MSE: {val_mse:.4f}, Edge: {val_edge:.4f}")
            
            # 保存最佳模型
            if val_total < best_loss:
                best_loss = val_total
                self.save_model("best_warmup_model.pth")
                print(f"  保存最佳模型 (Val Loss: {val_total:.4f})")
            
            # 每个epoch都保存权重
            epoch_checkpoint_path = f"epoch_{epoch}_checkpoint.pth"
            self.save_model(epoch_checkpoint_path)
            print(f"  Epoch {epoch} 权重已保存到: {epoch_checkpoint_path}")
        
        print("WARM-UP训练完成！")
    
    def save_model(self, filename):
        """保存模型"""
        save_path = Path("outputs/warmup_training") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'edge_head_state_dict': self.edge_head.state_dict(),
            'feature_adapter_state_dict': self.feature_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)
        
        print(f"模型已保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="第一版WARM-UP训练")
    parser.add_argument("--features-dir", type=str, required=True, help="特征NPZ文件目录")
    parser.add_argument("--images-dir", type=str, default=None, help="图像目录 (可选，不提供则使用随机图像)")
    parser.add_argument("--backbone", type=str, default="yolov8", choices=["yolov8", "repvit"], help="backbone类型")
    parser.add_argument("--head-type", type=str, default="simple", choices=["simple", "lightweight"], help="边缘头类型")
    parser.add_argument("--max-images", type=int, default=None, help="最大图像数量")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--mse-weight", type=float, default=1.0, help="MSE损失权重")
    parser.add_argument("--edge-weight", type=float, default=1.0, help="边缘损失权重")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="BCE损失权重")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="DICE损失权重")
    parser.add_argument("--pretrained", action="store_true", help="是否使用预训练权重")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="预训练权重文件路径")
    parser.add_argument("--freeze-backbone", action="store_true", help="是否冻结backbone")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'backbone': args.backbone,
        'head_type': args.head_type,
        'core_channels': 64,
        'learning_rate': args.lr,
        'mse_weight': args.mse_weight,
        'edge_weight': args.edge_weight,
        'bce_weight': args.bce_weight,
        'dice_weight': args.dice_weight,
        'pretrained': args.pretrained,
        'checkpoint_path': args.checkpoint_path,
        'freeze_backbone': args.freeze_backbone
    }
    
    print("=== 第一版WARM-UP训练 ===")
    print(f"特征目录: {args.features_dir}")
    print(f"图像目录: {args.images_dir or '未提供(使用随机图像)'}")
    print(f"Backbone: {args.backbone}")
    print(f"边缘头类型: {args.head_type}")
    print(f"最大图像数: {args.max_images or '全部'}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"损失权重 - MSE: {args.mse_weight}, Edge: {args.edge_weight}, BCE: {args.bce_weight}, DICE: {args.dice_weight}")
    
    # 创建数据集
    dataset = WarmUpDataset(
        features_dir=args.features_dir,
        images_dir=args.images_dir,
        max_images=args.max_images,
        input_size=1024
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建训练器
    trainer = WarmUpTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, args.epochs)
    
    print("WARM-UP训练完成！")


if __name__ == "__main__":
    main()

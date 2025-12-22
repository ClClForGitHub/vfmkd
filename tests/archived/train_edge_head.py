#!/usr/bin/env python3
"""
边缘头训练测试
使用SA-1B边缘图训练边缘头
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.models.heads.edge_head import UniversalEdgeHead
from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.backbones.repvit_backbone import RepViTBackbone


class SA1BEdgeDataset(Dataset):
    """SA-1B边缘数据集"""
    
    def __init__(self, images_dir, edges_dir, max_images=500, input_size=1024, edge_size=256):
        """
        Args:
            images_dir: 图像目录
            edges_dir: 边缘图目录
            max_images: 最大图像数量
            input_size: 输入图像尺寸 (1024x1024)
            edge_size: 边缘图尺寸 (256x256)
        """
        self.images_dir = Path(images_dir)
        self.edges_dir = Path(edges_dir)
        self.input_size = input_size
        self.edge_size = edge_size
        
        # 获取所有图像文件
        self.image_files = list(self.images_dir.glob("*.jpg"))[:max_images]
        
        # 过滤有效的图像-边缘对
        self.valid_pairs = []
        for img_file in self.image_files:
            edge_file = self.edges_dir / f"{img_file.stem}_edges_{edge_size}x{edge_size}.npy"
            if edge_file.exists():
                self.valid_pairs.append({
                    'image': img_file,
                    'edge': edge_file
                })
        
        print(f"SA-1B边缘数据集: {len(self.valid_pairs)} 个有效样本")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # 加载图像
        image = self._load_image(pair['image'])
        
        # 加载边缘图
        edge_map = np.load(pair['edge'])
        edge_tensor = torch.from_numpy(edge_map).float()
        
        return {
            'image': image,
            'edge_map': edge_tensor,
            'image_id': pair['image'].stem
        }
    
    def _load_image(self, image_path):
        """加载图像"""
        import cv2
        
        # 使用OpenCV加载图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸到1024x1024
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        # 转换为tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img


class EdgeHeadTrainer:
    """边缘头训练器"""
    
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
            head_type=config.get('head_type', 'simple')
        ).to(self.device)
        
        # 创建损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 创建优化器 - 包含backbone和边缘头参数
        all_params = list(self.backbone.parameters()) + list(self.edge_head.parameters())
        self.optimizer = optim.Adam(
            all_params,
            lr=config.get('learning_rate', 0.001)
        )
        
        print(f"边缘头参数: {sum(p.numel() for p in self.edge_head.parameters()):,}")
    
    def _create_backbone(self):
        """创建backbone"""
        backbone_type = self.config.get('backbone', 'yolov8')
        
        if backbone_type == 'yolov8':
            # YOLOv8配置 - 没有预训练权重，一起训练
            backbone_config = {
                'model_size': 's',  # 使用YOLOv8-s
                'pretrained': False,  # 没有预训练权重
                'freeze_backbone': False,  # 不冻结，一起训练
                'freeze_at': -1
            }
            backbone = YOLOv8Backbone(backbone_config)
        elif backbone_type == 'repvit':
            # RepViT配置 - 没有预训练权重，一起训练
            backbone_config = {
                'model_size': 'm1',
                'pretrained': False,  # 没有预训练权重
                'freeze_backbone': False,  # 不冻结，一起训练
                'freeze_at': -1
            }
            backbone = RepViTBackbone(backbone_config)
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")
        
        backbone = backbone.to(self.device)
        backbone.train()  # 训练模式，一起训练
        
        print(f"创建backbone: {backbone_type}")
        print(f"Backbone参数: {sum(p.numel() for p in backbone.parameters()):,}")
        return backbone
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.edge_head.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            edge_targets = batch['edge_map'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 提取特征 - backbone也参与训练
            features = self.backbone(images)
            # 使用P3特征（4倍下采样）
            p3_features = features[0]  # [B, 256, H, W]
            
            # 边缘预测
            edge_logits = self.edge_head(p3_features, backbone_name=self.config.get('backbone', 'yolov8'))
            
            # 将边缘预测上采样到256×256与真值对比
            if edge_logits.shape[2:] != edge_targets.shape[1:]:
                edge_logits = torch.nn.functional.interpolate(
                    edge_logits, 
                    size=edge_targets.shape[1:],  # 上采样到256×256
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 计算损失
            loss = self.criterion(edge_logits.squeeze(1), edge_targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_loss = total_loss / num_batches
        
        # 每个epoch保存一张边缘图
        self.save_edge_visualization(epoch, images, edge_logits, edge_targets)
        
        return avg_loss
    
    def save_edge_visualization(self, epoch, images, edge_logits, edge_targets):
        """保存边缘图可视化"""
        import cv2
        import numpy as np
        
        # 创建输出目录
        vis_dir = Path("outputs/edge_head_test/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 取第一个样本
        image = images[0].detach().cpu().permute(1, 2, 0).numpy()
        edge_pred = torch.sigmoid(edge_logits[0, 0]).detach().cpu().numpy()
        edge_target = edge_targets[0].detach().cpu().numpy()
        
        # 调整图像尺寸到256x256 (用于显示)
        image = cv2.resize(image, (256, 256))
        
        # 保存预测边缘图
        edge_pred_img = (edge_pred * 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"epoch_{epoch:02d}_pred.png"), edge_pred_img)
        
        # 保存真值边缘图
        edge_target_img = (edge_target * 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"epoch_{epoch:02d}_target.png"), edge_target_img)
        
        # 保存原图
        image_img = (image * 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"epoch_{epoch:02d}_image.png"), image_img)
        
        print(f"Epoch {epoch}: 边缘图已保存到 {vis_dir}")
    
    def validate(self, dataloader):
        """验证"""
        self.edge_head.eval()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                edge_targets = batch['edge_map'].to(self.device)
                
                # 提取特征
                features = self.backbone(images)
                p3_features = features[0]
                
                # 边缘预测
                edge_logits = self.edge_head(p3_features, backbone_name=self.config.get('backbone', 'yolov8'))
                
                # 将边缘预测上采样到256×256与真值对比
                if edge_logits.shape[2:] != edge_targets.shape[1:]:
                    edge_logits = torch.nn.functional.interpolate(
                        edge_logits, 
                        size=edge_targets.shape[1:],  # 上采样到256×256
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 计算损失
                loss = self.criterion(edge_logits.squeeze(1), edge_targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """训练边缘头"""
        print(f"开始训练边缘头，共 {epochs} 个epoch")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch + 1)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(f"best_edge_head.pth")
                print(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        print("训练完成！")
    
    def save_model(self, filename):
        """保存模型"""
        save_path = Path("outputs/edge_head_test") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'edge_head_state_dict': self.edge_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)
        
        print(f"模型已保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="边缘头训练测试")
    parser.add_argument("--images-dir", type=str, required=True, help="图像目录")
    parser.add_argument("--edges-dir", type=str, required=True, help="边缘图目录")
    parser.add_argument("--backbone", type=str, default="yolov8", choices=["yolov8", "repvit"], help="backbone类型")
    parser.add_argument("--head-type", type=str, default="simple", choices=["simple", "lightweight"], help="边缘头类型")
    parser.add_argument("--max-images", type=int, default=500, help="最大图像数量")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置
    config = {
        'backbone': args.backbone,
        'head_type': args.head_type,
        'core_channels': 64,
        'learning_rate': args.lr
    }
    
    print("=== 边缘头训练测试 ===")
    print(f"图像目录: {args.images_dir}")
    print(f"边缘图目录: {args.edges_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"边缘头类型: {args.head_type}")
    print(f"最大图像数: {args.max_images}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    
    # 创建数据集
    dataset = SA1BEdgeDataset(
        images_dir=args.images_dir,
        edges_dir=args.edges_dir,
        max_images=args.max_images,
        input_size=1024,  # 输入图像1024×1024
        edge_size=256     # 边缘图256×256
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
    trainer = EdgeHeadTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, args.epochs)
    
    print("边缘头训练测试完成！")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load trained model and generate visualizations
Compare different training methods
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import Dataset
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.edge_head import UniversalEdgeHead
from vfmkd.distillation.adapters import SimpleAdapter


class NPZDataset(Dataset):
    def __init__(self, features_dir: str, images_dir: str = None, max_images: int = None, input_size: int = 1024):
        self.features_dir = Path(features_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.input_size = input_size
        npz_files = sorted(self.features_dir.glob("*_features.npz"))
        if max_images:
            npz_files = npz_files[:max_images]
        self.valid_files = []
        for f in npz_files:
            try:
                data = np.load(f)
                if 'IMAGE_EMB_S16' in data and 'edge_256x256' in data:
                    self.valid_files.append(f)
            except Exception as e:
                print(f"  ⚠️  Skip file {f.name}: {e}")
                continue
        print(f"✅ Found {len(self.valid_files)} valid NPZ files")
    
    def __len__(self):
        return len(self.valid_files)
    
    def _load_real_image(self, image_id: str):
        """Load real image from datasets directory"""
        if self.images_dir is None:
            return torch.randn(3, self.input_size, self.input_size)
        
        # Try multiple possible file name formats
        possible_names = [
            f"sa_{image_id}.jpg",
            f"{image_id}.jpg",
            f"sa_{image_id}.png",
            f"{image_id}.png"
        ]
        
        for name in possible_names:
            image_file = self.images_dir / name
            if image_file.exists():
                try:
                    img = cv2.imread(str(image_file))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (self.input_size, self.input_size))
                    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                    return img_tensor
                except Exception as e:
                    continue
        
        # If not found, return random image
        print(f"  ⚠️  Image not found for ID {image_id}, using random")
        return torch.randn(3, self.input_size, self.input_size)
    
    def __getitem__(self, idx):
        npz_file = self.valid_files[idx]
        image_id = npz_file.stem.replace('_features', '').replace('sa_', '')
        data = np.load(npz_file)
        
        # Load real image
        image = self._load_real_image(image_id)
        
        return {
            'image': image,
            'teacher_features': torch.from_numpy(data['IMAGE_EMB_S16']).squeeze(0).float(),
            'edge_256x256': torch.from_numpy(data['edge_256x256']).float(),
            'image_id': image_id
        }


def load_model(checkpoint_path: str, device):
    """Load trained model"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create models
    backbone = YOLOv8Backbone({
        "model_size": "s",
        "pretrained": False,
        "freeze_backbone": False,
    }).to(device)
    
    edge_head = UniversalEdgeHead(
        core_channels=64,
        output_channels=1,
        head_type='simple',
        init_p=0.05
    ).to(device)
    
    feature_adapter = SimpleAdapter().to(device)
    
    # Load weights - use strict=False for dynamic layers
    backbone.load_state_dict(ckpt['backbone'])
    edge_head.load_state_dict(ckpt['edge_head'], strict=False)
    feature_adapter.load_state_dict(ckpt['feature_adapter'])
    
    # Set to eval mode
    backbone.eval()
    edge_head.eval()
    feature_adapter.eval()
    
    print("✅ Model loaded successfully")
    return backbone, edge_head, feature_adapter


def visualize_samples(backbone, edge_head, feature_adapter, dataset, output_dir, num_samples=10, device='cuda'):
    """Generate visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🖼️  Generating {num_samples} visualizations...")
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            teacher_features = sample['teacher_features'].unsqueeze(0).to(device)
            edge_gt = sample['edge_256x256']
            image_id = sample['image_id']
            
            # Forward pass
            features = backbone(image)
            s16_features = features[1]
            
            # Align features
            aligned_features = feature_adapter(s16_features, teacher_features)
            
            # Generate edge map
            edge_logits = edge_head(s16_features, backbone_name='yolov8')
            if edge_logits.shape[2:] != (256, 256):
                edge_pred = torch.nn.functional.interpolate(
                    edge_logits, size=(256, 256), mode="bilinear", align_corners=False
                )
            else:
                edge_pred = edge_logits
            edge_pred = torch.sigmoid(edge_pred[0, 0]).cpu().numpy()
            
            # P4 features
            p4_feat = aligned_features[0].cpu().numpy()
            p4_mean = p4_feat.mean(axis=0)
            p4_energy = np.sqrt((p4_feat ** 2).mean(axis=0))
            
            # Teacher features
            teacher_feat = teacher_features[0].cpu().numpy()
            teacher_mean = teacher_feat.mean(axis=0)
            
            # Original image
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            # Create visualization
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)
            
            # Row 1: Original, Edge GT, Edge Pred, Edge Overlay, Edge Error
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_np)
            ax0.set_title(f"Input Image (ID: {image_id})", fontsize=10)
            ax0.axis('off')
            
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(edge_gt.numpy(), cmap='gray', vmin=0, vmax=1)
            ax1.set_title("Edge GT (256x256)", fontsize=10)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(edge_pred, cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f"Edge Prediction\nmean={edge_pred.mean():.3f}", fontsize=10)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 3])
            img_resized = torch.nn.functional.interpolate(
                torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0),
                size=(256, 256), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0).numpy()
            ax3.imshow(img_resized)
            ax3.contour(edge_pred, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
            ax3.set_title("Edge Overlay (th=0.5)", fontsize=10)
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[0, 4])
            edge_diff = np.abs(edge_pred - edge_gt.numpy())
            im4 = ax4.imshow(edge_diff, cmap='hot', vmin=0, vmax=1)
            ax4.set_title(f"Edge Error\nMAE={edge_diff.mean():.3f}", fontsize=10)
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            ax4.axis('off')
            
            # Row 2: P4 Mean, P4 Energy, Teacher Mean, Feature Diff, Channel Grid
            ax5 = fig.add_subplot(gs[1, 0])
            im5 = ax5.imshow(p4_mean, cmap='viridis')
            ax5.set_title(f"Student P4 Mean (64x64)\nmean={p4_mean.mean():.3f}", fontsize=10)
            plt.colorbar(im5, ax=ax5, fraction=0.046)
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 1])
            im6 = ax6.imshow(p4_energy, cmap='hot')
            ax6.set_title(f"P4 Energy Map\nmax={p4_energy.max():.3f}", fontsize=10)
            plt.colorbar(im6, ax=ax6, fraction=0.046)
            ax6.axis('off')
            
            ax7 = fig.add_subplot(gs[1, 2])
            im7 = ax7.imshow(teacher_mean, cmap='viridis')
            ax7.set_title(f"Teacher SAM Mean\nmean={teacher_mean.mean():.3f}", fontsize=10)
            plt.colorbar(im7, ax=ax7, fraction=0.046)
            ax7.axis('off')
            
            ax8 = fig.add_subplot(gs[1, 3])
            feat_diff = np.abs(p4_mean - teacher_mean)
            im8 = ax8.imshow(feat_diff, cmap='hot')
            ax8.set_title(f"Feature Difference\nMAE={feat_diff.mean():.3f}", fontsize=10)
            plt.colorbar(im8, ax=ax8, fraction=0.046)
            ax8.axis('off')
            
            ax9 = fig.add_subplot(gs[1, 4])
            # First 16 channels in 4x4 grid
            n_show = min(16, p4_feat.shape[0])
            grid_size = 4
            channel_grid = np.zeros((grid_size * 16, grid_size * 16))
            for i in range(n_show):
                row, col = i // grid_size, i % grid_size
                ch_data = p4_feat[i]
                ch_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(ch_data).unsqueeze(0).unsqueeze(0),
                    size=(16, 16), mode='bilinear', align_corners=False
                )[0, 0].numpy()
                channel_grid[row*16:(row+1)*16, col*16:(col+1)*16] = ch_resized
            im9 = ax9.imshow(channel_grid, cmap='gray')
            ax9.set_title(f"First {n_show} Channels", fontsize=10)
            ax9.axis('off')
            
            # Save
            save_path = output_dir / f"{image_id}_visualization.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ {idx+1}/{num_samples}: {save_path.name}")
    
    print(f"\n🎉 Visualization complete! Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--features-dir", type=str, required=True, help="NPZ features directory")
    parser.add_argument("--images-dir", type=str, default=None, help="Original images directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    backbone, edge_head, feature_adapter = load_model(args.checkpoint, device)
    
    # Load dataset
    dataset = NPZDataset(args.features_dir, args.images_dir)
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Generate visualizations
    visualize_samples(
        backbone, edge_head, feature_adapter, 
        dataset, args.output_dir, 
        num_samples=args.num_samples,
        device=device
    )


if __name__ == "__main__":
    main()

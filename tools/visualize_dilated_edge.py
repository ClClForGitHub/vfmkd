"""
可视化膨胀后的边缘掩码
展示原始边缘、膨胀后边缘、真实图片叠加效果
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def dilate_edge(edge_map: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    使用MaxPool2d膨胀边缘图
    
    Args:
        edge_map: [H, W] 边缘图
        kernel_size: 膨胀核大小
        
    Returns:
        膨胀后的边缘图 [H, W]
    """
    # 添加batch和channel维度
    edge_4d = edge_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 使用MaxPool2d膨胀
    dilater = nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    
    dilated = dilater(edge_4d)
    return dilated.squeeze(0).squeeze(0)  # [H, W]


def visualize_dilated_edges(npz_dir: str, images_dir: str, output_dir: str, num_samples: int = 5):
    """
    可视化膨胀后的边缘
    
    Args:
        npz_dir: NPZ特征目录
        images_dir: 真实图片目录
        output_dir: 输出目录
        num_samples: 可视化样本数
    """
    npz_dir = Path(npz_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = list(npz_dir.glob("*_features.npz"))[:num_samples]
    
    print(f"Visualizing {len(npz_files)} samples...")
    
    for idx, npz_file in enumerate(npz_files):
        image_id = npz_file.stem.replace("_features", "")
        
        # 加载NPZ数据
        data = np.load(npz_file)
        edge_256 = torch.from_numpy(data["edge_256x256"]).float()
        
        # 加载真实图片
        img_candidates = [
            f"sa_{image_id}.jpg", f"{image_id}.jpg",
            f"sa_{image_id}.png", f"{image_id}.png",
        ]
        img_path = None
        for cand in img_candidates:
            p = images_dir / cand
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            print(f"[WARN] Image not found for {image_id}")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        
        # 膨胀边缘（不同核大小）
        edge_dilated_3 = dilate_edge(edge_256, kernel_size=3)
        edge_dilated_5 = dilate_edge(edge_256, kernel_size=5)
        edge_dilated_7 = dilate_edge(edge_256, kernel_size=7)
        
        # 转换为numpy
        edge_256_np = edge_256.numpy()
        edge_dilated_3_np = edge_dilated_3.numpy()
        edge_dilated_5_np = edge_dilated_5.numpy()
        edge_dilated_7_np = edge_dilated_7.numpy()
        
        # 创建可视化
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 第一行：边缘图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(edge_256_np, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Original Edge\n{edge_256_np.sum():.0f} pixels', fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(edge_dilated_3_np, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Dilated (k=3)\n{edge_dilated_3_np.sum():.0f} pixels', fontsize=12)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(edge_dilated_5_np, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Dilated (k=5)\n{edge_dilated_5_np.sum():.0f} pixels', fontsize=12)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(edge_dilated_7_np, cmap='gray', vmin=0, vmax=1)
        ax4.set_title(f'Dilated (k=7)\n{edge_dilated_7_np.sum():.0f} pixels', fontsize=12)
        ax4.axis('off')
        
        # 第二行：叠加到图片上
        ax5 = fig.add_subplot(gs[1, 0])
        overlay1 = img.copy()
        overlay1[edge_256_np > 0.5] = [255, 0, 0]  # 红色
        ax5.imshow(overlay1)
        ax5.set_title('Original Edge Overlay', fontsize=12)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        overlay2 = img.copy()
        overlay2[edge_dilated_3_np > 0.5] = [0, 255, 0]  # 绿色
        ax6.imshow(overlay2)
        ax6.set_title('Dilated (k=3) Overlay', fontsize=12)
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 2])
        overlay3 = img.copy()
        overlay3[edge_dilated_5_np > 0.5] = [0, 0, 255]  # 蓝色
        ax7.imshow(overlay3)
        ax7.set_title('Dilated (k=5) Overlay', fontsize=12)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 3])
        overlay4 = img.copy()
        overlay4[edge_dilated_7_np > 0.5] = [255, 255, 0]  # 黄色
        ax8.imshow(overlay4)
        ax8.set_title('Dilated (k=7) Overlay', fontsize=12)
        ax8.axis('off')
        
        fig.suptitle(f'Edge Dilation Visualization - {image_id}', fontsize=16, fontweight='bold')
        
        # 保存
        output_file = output_dir / f"{image_id}_dilated_edge.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [{idx+1}/{len(npz_files)}] Saved: {output_file.name}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    npz_dir = "VFMKD/outputs/features_v1_300"
    images_dir = "datasets/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0"
    output_dir = "VFMKD/outputs/testFGDFSD/dilated_edge_visualization"
    
    visualize_dilated_edges(npz_dir, images_dir, output_dir, num_samples=10)


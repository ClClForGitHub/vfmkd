#!/usr/bin/env python3
"""
可视化FGD/FSD前景/背景掩码，验证gt_adapter处理流程。
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfmkd.distillation.gt_adapter import build_fg_bg_from_ids, load_sa_json, _decode_union_mask_from_json


def visualize_mask_pipeline(json_path: str, feat_size: tuple = (64, 64), output_dir: str = "outputs/mask_vis", npz_path: Path = None):
    """
    完整可视化单个JSON的掩码处理流程：
    原图分割掩码 → 下采样至特征图 → 前景/背景权重 + 边缘图叠加
    
    Args:
        npz_path: NPZ文件路径，用于加载边缘图
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_path = Path(json_path)
    image_id = json_path.stem.replace("sa_", "")
    
    # 加载边缘图（如果提供了NPZ路径）
    # 使用与extract_features_v1.py完全一致的下采样方法
    edge_map = None
    if npz_path is not None and npz_path.exists():
        try:
            data = np.load(npz_path)
            # 优先使用edge_256x256，然后下采样到64x64
            if 'edge_256x256' in data:
                edge_256 = data['edge_256x256']  # (256, 256) uint8 {0, 1}
                
                # 【与extract_features_v1.py完全一致的处理】
                # 1. 转换为float32
                edge_float = edge_256.astype(np.float32)
                
                # 2. 使用cv2.resize + INTER_AREA下采样
                edge_64_float = cv2.resize(
                    edge_float,
                    (64, 64),
                    interpolation=cv2.INTER_AREA
                )
                
                # 3. 阈值化：只要 > 0 就设为 1
                edge_64_binary = (edge_64_float > 0).astype(np.uint8)
                
                # 转为torch tensor用于可视化
                edge_map = torch.from_numpy(edge_64_binary).float()
                print(f"  已加载边缘图: 256×256 → 64×64 (cv2.INTER_AREA + 阈值化), 唯一值={np.unique(edge_64_binary)}")
                
            elif 'edge_original' in data:
                edge_orig = data['edge_original']  # 原图尺寸 uint8 {0, 1}
                
                # 同样的处理方法
                edge_float = edge_orig.astype(np.float32)
                edge_64_float = cv2.resize(
                    edge_float,
                    (64, 64),
                    interpolation=cv2.INTER_AREA
                )
                edge_64_binary = (edge_64_float > 0).astype(np.uint8)
                edge_map = torch.from_numpy(edge_64_binary).float()
                print(f"  已加载边缘图: 原图 → 64×64 (cv2.INTER_AREA + 阈值化), 唯一值={np.unique(edge_64_binary)}")
        except Exception as e:
            print(f"  [WARN] 加载边缘图失败: {e}")
    
    # 1. 加载JSON并解码原图掩码
    obj = load_sa_json(json_path)
    img_info = obj.get("image", {})
    H_orig, W_orig = img_info.get("height", 1024), img_info.get("width", 1024)
    union_mask = _decode_union_mask_from_json(obj)  # (H_orig, W_orig) uint8 {0,1}
    
    if union_mask is None or union_mask.numel() == 0:
        print(f"[WARN] {image_id}: 无有效掩码")
        return
    
    print(f"[INFO] {image_id}: 原图尺寸={H_orig}x{W_orig}, 掩码形状={union_mask.shape}")
    
    # 2. 下采样到特征图尺寸
    Hf, Wf = feat_size
    m = union_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    fg_prob = torch.nn.functional.adaptive_avg_pool2d(m, (Hf, Wf)).squeeze(0).squeeze(0)  # (Hf,Wf) [0,1]
    fg_bin = (fg_prob > 0.5).float()  # 阈值化为二值
    
    # 3. 生成前景/背景权重（FGD风格）
    num_fg = torch.clamp(fg_bin.sum(), min=1)
    fg_area = fg_bin / num_fg  # area常数权重
    bg = 1.0 - fg_bin
    bg_sum = bg.sum()
    if bg_sum > 0:
        bg_normalized = bg / bg_sum
    else:
        bg_normalized = bg
    
    print(f"  下采样后: {Hf}x{Wf}, 前景像素={num_fg.item():.1f}/{Hf*Wf}, 占比={num_fg.item()/(Hf*Wf)*100:.1f}%")
    
    # 4. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Mask Pipeline: {image_id} | 原图{H_orig}x{W_orig} → 特征图{Hf}x{Wf}", fontsize=14)
    
    # 原图掩码（严格按照JSON标注，不假设语义）
    # union_mask: 严格按照JSON RLE解码结果，0和1的语义由标注决定
    axes[0, 0].imshow(union_mask.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f"原图RLE掩码 ({H_orig}x{W_orig})\n白=JSON中的1, 黑=JSON中的0")
    axes[0, 0].axis('off')
    
    # 下采样概率图（保持JSON定义） + 边缘图叠加
    axes[0, 1].imshow(fg_prob.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    if edge_map is not None:
        # 叠加边缘图：使用亮绿色，设置透明度
        edge_threshold = 0.3  # 边缘阈值
        edge_mask = (edge_map > edge_threshold).cpu().numpy()
        # 创建绿色RGBA图像作为边缘高亮
        from matplotlib.colors import LinearSegmentedColormap
        axes[0, 1].contour(edge_mask, levels=[0.5], colors='lime', linewidths=2, alpha=0.9)
    axes[0, 1].set_title(f"下采样概率图+边缘 ({Hf}x{Wf})\nadaptive_avg_pool2d\n红=JSON中1, 蓝=JSON中0, 绿=边缘")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    axes[0, 1].axis('off')
    
    # 阈值化二值图（prob > 0.5） + 边缘图叠加
    axes[0, 2].imshow(fg_bin.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    if edge_map is not None:
        # 叠加边缘图轮廓
        axes[0, 2].contour(edge_mask, levels=[0.5], colors='lime', linewidths=2, alpha=0.9)
    axes[0, 2].set_title(f"阈值化二值图+边缘 ({Hf}x{Wf})\nprob > 0.5\n白=JSON中1, 黑=JSON中0, 绿=边缘")
    axes[0, 2].axis('off')
    
    # FGD权重图1：权重化"JSON中为1的区域"
    axes[1, 0].imshow(fg_area.cpu().numpy(), cmap='hot', vmin=0, vmax=fg_area.max().item())
    axes[1, 0].set_title(f"权重图1 ({Hf}x{Wf})\narea=1/{num_fg.item():.0f}\n黄=JSON中1的区域(有权重)")
    cbar1 = plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046)
    cbar1.set_label('Area weight')
    axes[1, 0].axis('off')
    
    # 反转二值图
    axes[1, 1].imshow(bg.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f"反转二值图 ({Hf}x{Wf})\n1 - (prob>0.5)\n白=JSON中0的区域")
    axes[1, 1].axis('off')
    
    # FGD权重图2：权重化"JSON中为0的区域"
    axes[1, 2].imshow(bg_normalized.cpu().numpy(), cmap='viridis', vmin=0, vmax=bg_normalized.max().item())
    axes[1, 2].set_title(f"权重图2 ({Hf}x{Wf})\nnormalized\n黄=JSON中0的区域(有权重)")
    cbar2 = plt.colorbar(axes[1, 2].images[0], ax=axes[1, 2], fraction=0.046)
    cbar2.set_label('Normalized prob')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_file = output_path / f"{image_id}_mask_pipeline.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"  已保存: {out_file}")
    plt.close()


def visualize_one_sample(npz_file: Path, json_dir: Path, feat_size: tuple = (64, 64), output_path: Path = None):
    """
    从NPZ文件提取image_id，然后可视化对应的JSON掩码处理流程（包含边缘图）
    """
    # 从NPZ文件名提取image_id: sa_10000_features.npz -> 10000
    image_id = npz_file.stem.replace("_features", "").replace("sa_", "")
    json_file = json_dir / f"sa_{image_id}.json"
    
    if not json_file.exists():
        print(f"[WARN] 跳过ID={image_id}: 找不到JSON文件 {json_file}")
        return
    
    # 调用可视化函数，传递NPZ路径用于加载边缘图
    if output_path is None:
        output_path = npz_file.parent.parent / "mask_vis"
    visualize_mask_pipeline(json_file, feat_size=feat_size, output_dir=str(output_path), npz_path=npz_file)


def batch_visualize_from_npz_dir(npz_dir: str, json_dir: str, max_samples: int = 5, feat_size: tuple = (64, 64), target_ids: list = None):
    """
    从NPZ目录批量可视化样本的掩码处理流程
    
    Args:
        target_ids: 指定要可视化的图片ID列表；如果为None则取前max_samples个
    """
    npz_dir = Path(npz_dir)
    json_dir = Path(json_dir)
    output_path = npz_dir.parent / "mask_vis"
    output_path.mkdir(exist_ok=True, parents=True)
    
    if target_ids is not None:
        # 可视化指定ID的图片
        print(f"[INFO] 将可视化指定的{len(target_ids)}个样本: {target_ids}")
        print(f"  NPZ目录: {npz_dir}")
        print(f"  JSON目录: {json_dir}")
        print(f"  输出目录: {output_path}")
        print(f"  特征图尺寸: {feat_size}")
        print()
        
        for image_id in target_ids:
            # 查找对应的NPZ文件
            npz_pattern = npz_dir / f"sa_{image_id}_features.npz"
            if not npz_pattern.exists():
                print(f"[WARN] 跳过ID={image_id}: 找不到NPZ文件 {npz_pattern}")
                continue
            
            visualize_one_sample(npz_pattern, json_dir, feat_size=feat_size, output_path=output_path)
    else:
        # 可视化前N个样本
        npz_files = sorted(npz_dir.glob("*_features.npz"))[:max_samples]
        
        print(f"[INFO] 将可视化{len(npz_files)}个样本的掩码处理流程")
        print(f"  NPZ目录: {npz_dir}")
        print(f"  JSON目录: {json_dir}")
        print(f"  输出目录: {output_path}")
        print(f"  特征图尺寸: {feat_size}")
        print()
        
        for npz_file in npz_files:
            visualize_one_sample(npz_file, json_dir, feat_size=feat_size, output_path=output_path)


def main():
    # 配置路径
    npz_dir = r"C:\AiBuild\paper\detect\EdgeSAM-master\VFMKD\outputs\features_v1_300"
    json_dir = r"C:\AiBuild\paper\detect\EdgeSAM-master\datasets\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0"
    
    # 生成前10张图片
    npz_dir_path = Path(npz_dir)
    npz_files = sorted(npz_dir_path.glob("sa_*_features.npz"))[:10]
    target_ids = [f.stem.replace("_features", "").replace("sa_", "") for f in npz_files]
    
    print(f"将处理前10张图片（cv2.INTER_AREA边缘处理 + 绿色轮廓叠加）: {target_ids}")
    
    batch_visualize_from_npz_dir(
        npz_dir,
        json_dir,
        feat_size=(64, 64),
        target_ids=target_ids
    )
    
    output_path = Path(npz_dir).parent / "mask_vis"
    print(f"\n✅ 可视化完成！结果保存在: {output_path}")
    print(f"   生成了 {len(target_ids)} 张掩码处理流程图（包含边缘图叠加）")
    print(f"   边缘图用亮绿色轮廓线标识在热力图和二值图上")


if __name__ == "__main__":
    main()


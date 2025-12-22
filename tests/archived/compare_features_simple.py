#!/usr/bin/env python3
"""
简化版：对比 RepViT+Adapter 特征和掩码
不使用SAM2官方编码器，只对比我们自己的模型生成的特征和掩码
"""
import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

# Stub heavy deps
import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear
        sys.modules[_mod] = stub_mod

from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def visualize_feature_map(feat: torch.Tensor, title: str, ax):
    """Visualize feature map - show average across all channels"""
    # feat: (B, C, H, W)
    feat_np = feat[0].cpu().numpy()  # (C, H, W)
    
    # Calculate mean across all channels
    feat_mean = feat_np.mean(axis=0)  # (H, W)
    
    # Normalize to 0-1
    feat_min = feat_mean.min()
    feat_max = feat_mean.max()
    if feat_max > feat_min:
        feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
    else:
        feat_norm = feat_mean
    
    im = ax.imshow(feat_norm, cmap='viridis')
    ax.set_title(f"{title}\nShape: {feat.shape}\nMean: {feat.mean().item():.4f}, Std: {feat.std().item():.4f}\nRange: [{feat.min().item():.2f}, {feat.max().item():.2f}]", fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)


def visualize_mask(mask_logits: torch.Tensor, title: str, ax, binary_threshold=0.0):
    """Visualize segmentation mask - input is logits
    
    Args:
        mask_logits: Raw logits from mask decoder
        title: Title for the plot
        ax: Matplotlib axis
        binary_threshold: Threshold for binarization (applied to logits directly)
            - If 0.0: threshold at logit=0 (sigmoid=0.5)
            - If None: show sigmoid probabilities as grayscale
    """
    mask_np = mask_logits[0, 0].cpu().numpy()  # (H, W)
    
    if binary_threshold is not None:
        # Binary visualization: logit > threshold -> white, else black
        mask_binary = (mask_np > binary_threshold).astype(np.uint8) * 255
        ax.imshow(mask_binary, cmap='gray', vmin=0, vmax=255)
        pos_pixels = np.sum(mask_binary > 128)
        ax.set_title(f"{title}\nPositive pixels: {pos_pixels}", fontsize=9)
    else:
        # Probability visualization with contrast stretch
        mask_probs = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        # Stretch to full range for better visibility
        p_min, p_max = mask_probs.min(), mask_probs.max()
        if p_max > p_min:
            mask_stretched = (mask_probs - p_min) / (p_max - p_min)
        else:
            mask_stretched = mask_probs
        ax.imshow(mask_stretched, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{title}\nProb range: [{p_min:.3f}, {p_max:.3f}]", fontsize=9)
    
    ax.axis('off')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='对比特征和掩码')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--weights', type=str, required=True, help='训练得到的权重路径')
    parser.add_argument('--teacher_npz', type=str, default='', help='教师特征NPZ文件（可选）')
    parser.add_argument('--sam2_checkpoint', type=str, default='weights/sam2.1_hiera_base_plus.pt', help='SAM2官方权重')
    parser.add_argument('--point', type=float, nargs=2, default=[512, 512], help='点击坐标')
    parser.add_argument('--label', type=int, default=1, help='点击标签')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/feature_comparison.png')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载图片
    print(f"[INFO] 加载图片: {args.image}")
    img_pil = Image.open(args.image).convert('RGB').resize((1024, 1024))
    x = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    print(f"[INFO] 图片 shape: {x.shape}")
    
    # ========== 1. RepViT + Adapter ==========
    print("\n" + "="*80)
    print("[INFO] 初始化 RepViT + Adapter...")
    
    # 初始化 backbone
    config = {'arch': 'm1', 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None}
    backbone = RepViTBackbone(config).to(device).eval()
    
    # 初始化 adapter
    adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256).to(device).eval()
    
    # 加载权重
    print(f"[INFO] 加载权重: {args.weights}")
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['backbone'], strict=True)
    adapter.load_state_dict(ckpt['adapter'], strict=True)
    print("[INFO] 权重加载成功!")
    
    # 前向传播
    print("[INFO] RepViT backbone 前向传播...")
    backbone_feat = backbone(x)
    print(f"[INFO] Backbone 输出 shape: {backbone_feat.shape}")
    print(f"[INFO] Backbone 特征统计: mean={backbone_feat.mean().item():.4f}, std={backbone_feat.std().item():.4f}, "
          f"min={backbone_feat.min().item():.4f}, max={backbone_feat.max().item():.4f}")
    
    print("[INFO] Adapter 前向传播...")
    adapter_feat = adapter(backbone_feat)
    print(f"[INFO] Adapter 输出 shape: {adapter_feat.shape}")
    print(f"[INFO] Adapter 特征统计: mean={adapter_feat.mean().item():.4f}, std={adapter_feat.std().item():.4f}, "
          f"min={adapter_feat.min().item():.4f}, max={adapter_feat.max().item():.4f}")
    
    # ========== 2. 加载教师特征（如果提供） ==========
    teacher_feat = None
    if args.teacher_npz and Path(args.teacher_npz).exists():
        print("\n" + "="*80)
        print(f"[INFO] 加载教师特征: {args.teacher_npz}")
        teacher_data = np.load(args.teacher_npz)
        if 'IMAGE_EMB_S16' in teacher_data:
            teacher_feat = torch.from_numpy(teacher_data['IMAGE_EMB_S16']).float().to(device)
        elif 'image_embedding' in teacher_data:
            teacher_feat = torch.from_numpy(teacher_data['image_embedding']).float().to(device)
        
        if teacher_feat is not None:
            # 去掉多余维度
            if teacher_feat.dim() == 5:
                teacher_feat = teacher_feat.squeeze(1)  # (B, 1, C, H, W) -> (B, C, H, W)
            print(f"[INFO] 教师特征 shape: {teacher_feat.shape}")
            print(f"[INFO] 教师特征统计: mean={teacher_feat.mean().item():.4f}, std={teacher_feat.std().item():.4f}, "
                  f"min={teacher_feat.min().item():.4f}, max={teacher_feat.max().item():.4f}")
    
    # ========== 3. 加载 SAM2 预训练模型（包括 Prompt Encoder 和 Mask Decoder） ==========
    print("\n" + "="*80)
    print("[INFO] 加载 SAM2 预训练模型...")
    
    # 根据checkpoint文件名选择配置
    if 'base_plus' in args.sam2_checkpoint:
        config_name = 'sam2.1/sam2.1_hiera_b+.yaml'
    elif 'large' in args.sam2_checkpoint:
        config_name = 'sam2.1/sam2.1_hiera_l.yaml'
    elif 'small' in args.sam2_checkpoint:
        config_name = 'sam2.1/sam2.1_hiera_s.yaml'
    else:
        config_name = 'sam2.1/sam2.1_hiera_t.yaml'
    
    print(f"[INFO] 使用配置: {config_name}")
    print(f"[INFO] 加载权重: {args.sam2_checkpoint}")
    
    # 检查权重文件
    checkpoint_path = Path(args.sam2_checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] 权重文件不存在: {checkpoint_path}")
        return
    
    # 设置Hydra配置搜索路径
    sam2_config_dir = _ROOT / "vfmkd" / "sam2" / "sam2" / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # 构建SAM2模型
    with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
        sam2_model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(device)
        )
    
    sam2_model.eval()
    print(f"[INFO] SAM2模型加载成功!")
    
    # 使用SAM2内置的prompt encoder和mask decoder
    prompt_encoder = sam2_model.sam_prompt_encoder
    mask_decoder = sam2_model.sam_mask_decoder
    
    # 检查是否使用高分辨率特征
    use_high_res = getattr(sam2_model, 'use_high_res_features_in_sam', False)
    print(f"[INFO] 使用高分辨率特征: {use_high_res}")
    
    # 准备点击提示
    point_coords = torch.tensor([[args.point]], dtype=torch.float32, device=device)  # (B, N, 2)
    point_labels = torch.tensor([[args.label]], dtype=torch.int32, device=device)  # (B, N)
    print(f"[INFO] 点击坐标: {args.point}, 标签: {args.label}")
    
    # ========== 4. 生成掩码 ==========
    print("\n" + "="*80)
    print("[INFO] 使用 RepViT + Adapter 生成掩码...")
    
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    
    # 根据模型配置决定是否传递高分辨率特征
    if use_high_res:
        # 暂时禁用高分辨率特征，直接修改mask_decoder的配置
        print("[WARN] 模型配置为使用高分辨率特征，但RepViT不提供，临时禁用")
        mask_decoder.use_high_res_features = False
        high_res_feat_arg = None
    else:
        high_res_feat_arg = None
    
    low_res_masks, iou_predictions, _, _ = mask_decoder(
        image_embeddings=adapter_feat,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
        high_res_features=high_res_feat_arg,
    )
    
    print(f"[INFO] RepViT+Adapter 掩码 shape: {low_res_masks.shape}")
    print(f"[INFO] RepViT+Adapter IOU 预测: {iou_predictions[0].cpu().numpy()}")
    
    # ========== 5. 使用 SAM2 原生特征生成掩码 ==========
    if teacher_feat is not None:
        print("\n" + "="*80)
        print("[INFO] 使用 SAM2 原生特征生成掩码（对比基准）...")
        
        low_res_masks_sam2, iou_predictions_sam2, _, _ = mask_decoder(
            image_embeddings=teacher_feat,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_feat_arg,
        )
        
        print(f"[INFO] SAM2 原生掩码 shape: {low_res_masks_sam2.shape}")
        print(f"[INFO] SAM2 原生 IOU 预测: {iou_predictions_sam2[0].cpu().numpy()}")
    else:
        low_res_masks_sam2 = None
        iou_predictions_sam2 = None
    
    # ========== 6. 可视化 ==========
    print("\n" + "="*80)
    print("[INFO] 生成可视化...")
    
    # Determine layout: 3 rows x 4 columns if we have SAM2 native features
    if teacher_feat is not None and low_res_masks_sam2 is not None:
        fig = plt.figure(figsize=(20, 15))
        n_cols = 4
        n_rows = 3
    else:
        fig = plt.figure(figsize=(16, 10))
        n_cols = 4
        n_rows = 2
    
    # Row 1: Original image + Feature visualizations
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax1.imshow(img_pil)
    ax1.set_title("Original Image", fontsize=10)
    ax1.axis('off')
    ax1.plot(args.point[0], args.point[1], 'r*', markersize=15)
    
    ax2 = plt.subplot(n_rows, n_cols, 2)
    visualize_feature_map(backbone_feat, "RepViT Backbone Output", ax2)
    
    ax3 = plt.subplot(n_rows, n_cols, 3)
    visualize_feature_map(adapter_feat, "Adapter Output", ax3)
    
    if teacher_feat is not None:
        ax4 = plt.subplot(n_rows, n_cols, 4)
        visualize_feature_map(teacher_feat, "SAM2 Teacher Features", ax4)
    
    # Row 2: RepViT+Adapter masks
    for i in range(3):
        ax = plt.subplot(n_rows, n_cols, n_cols + 1 + i)
        mask = F.interpolate(low_res_masks[:, i:i+1], size=(1024, 1024), mode='bilinear', align_corners=False)
        mask_binary = (mask > 0).float()
        visualize_mask(mask_binary, f"RepViT+Adapter Mask {i}\nIOU: {iou_predictions[0, i].item():.3f}", ax)
    
    # Row 2, Column 4: Best RepViT+Adapter mask
    best_idx_repvit = iou_predictions[0].argmax().item()
    ax = plt.subplot(n_rows, n_cols, n_cols + 4)
    mask = F.interpolate(low_res_masks[:, best_idx_repvit:best_idx_repvit+1], size=(1024, 1024), mode='bilinear', align_corners=False)
    mask_binary = (mask > 0).float()
    visualize_mask(mask_binary, f"RepViT+Adapter Best\n(Mask {best_idx_repvit}, IOU: {iou_predictions[0, best_idx_repvit].item():.3f})", ax)
    
    # Row 3: SAM2 Native masks (if available)
    if low_res_masks_sam2 is not None:
        for i in range(3):
            ax = plt.subplot(n_rows, n_cols, 2 * n_cols + 1 + i)
            mask = F.interpolate(low_res_masks_sam2[:, i:i+1], size=(1024, 1024), mode='bilinear', align_corners=False)
            mask_binary = (mask > 0).float()
            visualize_mask(mask_binary, f"SAM2 Native Mask {i}\nIOU: {iou_predictions_sam2[0, i].item():.3f}", ax)
        
        # Row 3, Column 4: Best SAM2 Native mask
        best_idx_sam2 = iou_predictions_sam2[0].argmax().item()
        ax = plt.subplot(n_rows, n_cols, 2 * n_cols + 4)
        mask = F.interpolate(low_res_masks_sam2[:, best_idx_sam2:best_idx_sam2+1], size=(1024, 1024), mode='bilinear', align_corners=False)
        mask_binary = (mask > 0).float()
        visualize_mask(mask_binary, f"SAM2 Native Best\n(Mask {best_idx_sam2}, IOU: {iou_predictions_sam2[0, best_idx_sam2].item():.3f})", ax)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 可视化已保存到: {args.output}")
    
    # Print feature differences and mask comparison
    if teacher_feat is not None:
        print("\n" + "="*80)
        print("[INFO] Feature Difference Statistics:")
        feat_diff = (adapter_feat - teacher_feat).abs()
        print(f"  Absolute Diff: mean={feat_diff.mean().item():.4f}, std={feat_diff.std().item():.4f}, "
              f"max={feat_diff.max().item():.4f}")
        
        # Cosine similarity
        adapter_flat = F.normalize(adapter_feat.flatten(1), dim=1)
        teacher_flat = F.normalize(teacher_feat.flatten(1), dim=1)
        cosine_sim = (adapter_flat * teacher_flat).sum(dim=1).mean()
        print(f"  Cosine Similarity: {cosine_sim.item():.4f}")
        
        # Mask comparison
        if low_res_masks_sam2 is not None:
            print("\n" + "="*80)
            print("[INFO] Mask Quality Comparison:")
            print(f"  RepViT+Adapter Best IOU: {iou_predictions[0].max().item():.4f}")
            print(f"  SAM2 Native Best IOU: {iou_predictions_sam2[0].max().item():.4f}")
            print(f"  IOU Gap: {(iou_predictions_sam2[0].max() - iou_predictions[0].max()).item():.4f}")


if __name__ == '__main__':
    main()


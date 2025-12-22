#!/usr/bin/env python3
"""
对比 RepViT+Adapter 和 SAM2 官方编码器的特征和分割结果
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
from sam2.modeling.sam.prompt_encoder import PromptEncoder as PromptEncoderSAM2
from sam2.modeling.sam.mask_decoder import MaskDecoder as MaskDecoderSAM2
from sam2.modeling.sam.transformer import TwoWayTransformer as TwoWayTransformerSAM2
from sam2.build_sam import build_sam2


def visualize_feature_map(feat: torch.Tensor, title: str, ax):
    """可视化特征图 - 显示所有通道的平均值"""
    # feat: (B, C, H, W)
    feat_np = feat[0].cpu().numpy()  # (C, H, W)
    
    # 计算所有通道的均值
    feat_mean = feat_np.mean(axis=0)  # (H, W)
    
    # 归一化到 0-1
    feat_min = feat_mean.min()
    feat_max = feat_mean.max()
    if feat_max > feat_min:
        feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
    else:
        feat_norm = feat_mean
    
    im = ax.imshow(feat_norm, cmap='viridis')
    ax.set_title(f"{title}\nShape: {feat.shape}\nMean: {feat.mean().item():.4f}, Std: {feat.std().item():.4f}\nRange: [{feat.min().item():.2f}, {feat.max().item():.2f}]")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)


def visualize_mask(mask: torch.Tensor, title: str, ax):
    """可视化掩码"""
    mask_np = mask[0, 0].cpu().numpy()  # (H, W)
    ax.imshow(mask_np, cmap='gray')
    ax.set_title(title)
    ax.axis('off')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='对比特征和掩码')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--weights', type=str, required=True, help='训练得到的权重路径')
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
    ckpt = torch.load(args.weights, map_location='cpu')
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
    
    # ========== 2. SAM2 官方编码器 ==========
    print("\n" + "="*80)
    print("[INFO] 初始化 SAM2 官方模型...")
    
    # 根据checkpoint文件名选择配置
    if 'base_plus' in args.sam2_checkpoint:
        config_name = 'sam2_hiera_b+'
    elif 'large' in args.sam2_checkpoint:
        config_name = 'sam2_hiera_l'
    elif 'small' in args.sam2_checkpoint:
        config_name = 'sam2_hiera_s'
    else:
        config_name = 'sam2_hiera_t'
    
    print(f"[INFO] 使用配置: {config_name}")
    print(f"[INFO] 加载权重: {args.sam2_checkpoint}")
    
    # 使用build_sam2加载完整模型
    sam2_model = build_sam2(
        config_file=config_name,
        ckpt_path=args.sam2_checkpoint if Path(args.sam2_checkpoint).exists() else None,
        device=device,
        mode='eval'
    )
    
    # 前向传播
    print("[INFO] SAM2 图像编码器前向传播...")
    sam2_backbone_out = sam2_model.forward_image(x)
    sam2_feat = sam2_backbone_out['vision_features']
    print(f"[INFO] SAM2 输出 shape: {sam2_feat.shape}")
    print(f"[INFO] SAM2 特征统计: mean={sam2_feat.mean().item():.4f}, std={sam2_feat.std().item():.4f}, "
          f"min={sam2_feat.min().item():.4f}, max={sam2_feat.max().item():.4f}")
    
    # ========== 3. 使用 SAM2 模型内置的 Prompt Encoder 和 Mask Decoder ==========
    print("\n" + "="*80)
    print("[INFO] 使用 SAM2 内置的 Prompt Encoder 和 Mask Decoder...")
    
    prompt_encoder = sam2_model.sam_prompt_encoder
    mask_decoder = sam2_model.sam_mask_decoder
    
    # 准备点击提示
    point_coords = torch.tensor([[args.point]], dtype=torch.float32, device=device)  # (B, N, 2)
    point_labels = torch.tensor([[args.label]], dtype=torch.int32, device=device)  # (B, N)
    print(f"[INFO] 点击坐标: {args.point}, 标签: {args.label}")
    
    # ========== 4. 生成掩码 - RepViT + Adapter ==========
    print("\n" + "="*80)
    print("[INFO] 使用 RepViT + Adapter 生成掩码...")
    
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    
    low_res_masks_repvit, iou_predictions_repvit, _, _ = mask_decoder(
        image_embeddings=adapter_feat,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
    )
    
    print(f"[INFO] RepViT 掩码 shape: {low_res_masks_repvit.shape}")
    print(f"[INFO] RepViT IOU 预测: {iou_predictions_repvit[0].cpu().numpy()}")
    
    # ========== 5. 生成掩码 - SAM2 官方 ==========
    print("\n" + "="*80)
    print("[INFO] 使用 SAM2 官方编码器生成掩码...")
    
    low_res_masks_sam2, iou_predictions_sam2, _, _ = mask_decoder(
        image_embeddings=sam2_feat,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
    )
    
    print(f"[INFO] SAM2 掩码 shape: {low_res_masks_sam2.shape}")
    print(f"[INFO] SAM2 IOU 预测: {iou_predictions_sam2[0].cpu().numpy()}")
    
    # ========== 6. 可视化 ==========
    print("\n" + "="*80)
    print("[INFO] 生成可视化...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 第一行：原图 + 特征可视化
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_pil)
    ax1.set_title("原始图片")
    ax1.axis('off')
    ax1.plot(args.point[0], args.point[1], 'r*', markersize=20)
    
    ax2 = plt.subplot(3, 4, 2)
    visualize_feature_map(backbone_feat, "RepViT Backbone 输出", ax2)
    
    ax3 = plt.subplot(3, 4, 3)
    visualize_feature_map(adapter_feat, "Adapter 输出", ax3)
    
    ax4 = plt.subplot(3, 4, 4)
    visualize_feature_map(sam2_feat, "SAM2 官方编码器输出", ax4)
    
    # 第二行：RepViT+Adapter 生成的3个掩码
    for i in range(3):
        ax = plt.subplot(3, 4, 5 + i)
        mask = F.interpolate(low_res_masks_repvit[:, i:i+1], size=(1024, 1024), mode='bilinear', align_corners=False)
        mask_binary = (mask > 0).float()
        visualize_mask(mask_binary, f"RepViT 掩码 {i}\nIOU: {iou_predictions_repvit[0, i].item():.3f}", ax)
    
    # 第三行：SAM2 官方生成的3个掩码
    for i in range(3):
        ax = plt.subplot(3, 4, 9 + i)
        mask = F.interpolate(low_res_masks_sam2[:, i:i+1], size=(1024, 1024), mode='bilinear', align_corners=False)
        mask_binary = (mask > 0).float()
        visualize_mask(mask_binary, f"SAM2 官方掩码 {i}\nIOU: {iou_predictions_sam2[0, i].item():.3f}", ax)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 可视化已保存到: {args.output}")
    
    # 打印特征差异
    print("\n" + "="*80)
    print("[INFO] 特征差异统计:")
    feat_diff = (adapter_feat - sam2_feat).abs()
    print(f"  绝对差异: mean={feat_diff.mean().item():.4f}, std={feat_diff.std().item():.4f}, "
          f"max={feat_diff.max().item():.4f}")
    
    # 余弦相似度
    adapter_flat = F.normalize(adapter_feat.flatten(1), dim=1)
    sam2_flat = F.normalize(sam2_feat.flatten(1), dim=1)
    cosine_sim = (adapter_flat * sam2_flat).sum(dim=1).mean()
    print(f"  余弦相似度: {cosine_sim.item():.4f}")


if __name__ == '__main__':
    main()


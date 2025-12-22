#!/usr/bin/env python3
"""
独立统一测试脚本：在固定测试集上，使用统一指标（Feature MSE/MAE、Cosine、Edge BCE+Dice）
对已训练好的模型进行公平对比。

严格复用当前训练脚本的关键逻辑：
- 复用数据集类 NPZWithImageIdDataset（真实图片+NPZ严格配对）
- 复用模型组件创建方式（YOLOv8Backbone、UniversalEdgeHead、SimpleAdapter）
- 复用统一指标评估实现 validate_unified_metrics 的核心计算流程

用法示例：
python tools/core/unified_model_test.py \
  --features-dir /home/team/zouzhiyuan/dataset/sa1b/extracted \
  --images-dir /home/team/zouzhiyuan/dataset/sa1b \
  --checkpoints \
    outputs/distill_single_test_MSE/xxx/models/epoch_2_model.pth \
    outputs/distill_single_test_FGD/xxx/models/epoch_2_model.pth \
  --names MSE FGD \
  --batch-size 4 \
  --output outputs/unified_model_test.txt
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Subset

# 路径：将项目根目录加入sys.path，保持与训练脚本一致
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接导入，因为已经在同一个目录下
# 使用importlib导入，避免路径问题
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_distill_single_test",
    Path(__file__).parent / "train_distill_single_test.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
NPZWithImageIdDataset = train_module.NPZWithImageIdDataset
DistillSingleTester = train_module.DistillSingleTester


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_features_dir: Path,
    test_images_dir: Path,
    batch_size: int = 4,
    max_images: int = None,
    gt_json_dir: Path | None = None,
) -> dict:
    """
    加载checkpoint，构建与训练一致的模型组件，然后在固定测试集上用统一指标评估。
    使用固定的1000张测试集（test目录），或从训练集中选择指定数量的样本。
    返回：{"mse":..., "mae":..., "cosine_sim":..., "edge_loss":...}
    """
    # 使用测试集或训练集（可通过max_images限制数量）
    test_dataset = NPZWithImageIdDataset(str(test_features_dir), str(test_images_dir), max_images=max_images, input_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 构建与训练一致的模型容器（不做训练，只用于评估）
    config = {
        "backbone": "yolov8",
        # 与训练一致的edge损失设置（统一指标使用BCE+Dice，不启用掩码/pos_weight）
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "edge_mask_kernel_size": 3,
        "use_pos_weight": False,
        "enable_edge_mask_progressive": False,
    }
    runner = DistillSingleTester(config)

    # 加载checkpoint（严格对应键名，与训练脚本保持一致）
    ckpt = torch.load(str(checkpoint_path), map_location=runner.device)
    
    # 加载backbone
    runner.backbone.load_state_dict(ckpt["backbone"])
    
    # 加载edge_head（简化版，固定256通道输入）
    runner.edge_head.load_state_dict(ckpt["edge_head"], strict=False)
    
    # 加载静态适配器（直接load_state_dict，兼容旧V1键名，strict=False）
    if "edge_adapter" in ckpt:
        runner.edge_adapter.load_state_dict(ckpt["edge_adapter"], strict=False)
    else:
        print("[WARNING] checkpoint中没有edge_adapter（可能是旧格式），已跳过")
    if "feature_adapter" in ckpt:
        runner.feature_adapter.load_state_dict(ckpt["feature_adapter"], strict=False)

    # 使用训练脚本内的统一指标实现，保持计算细节一致
    metrics = runner.validate_unified_metrics(test_loader)

    # 可选：补充计算FGD与FSD的loss（用于横向对比不同蒸馏指标的倾向）
    # 优先从NPZ读取预计算权重（与训练保持一致），缺失时且提供了gt_json_dir时再回退JSON
    metrics_fgd, metrics_fsd = None, None
    # 延迟导入损失与gt/npz工具
    from vfmkd.distillation.losses.fgd_loss import FGDLoss
    from vfmkd.distillation.losses.fsd_loss import FSDLikeLoss
    from vfmkd.distillation.gt_adapter import build_fg_bg_from_ids
    # NPZ优先导入（可能不可用时再回退JSON）
    try:
        from vfmkd.distillation.gt_adapter import load_weights_from_npz, load_edge_maps_from_npz
    except Exception:
        load_weights_from_npz = None
        load_edge_maps_from_npz = None

    fgd_loss_fn = FGDLoss(alpha_fg=0.001, beta_bg=0.0005, alpha_edge=0.002, gamma_mask=0.0, lambda_rela=0.0, temperature=1.0).to(runner.device)
    fsd_loss_fn = FSDLikeLoss(weight_fg=1.0, weight_bg=0.2, temperature=1.0, gamma_mask=0.0, lambda_rela=0.0, gaussian_from_mask=False, gaussian_mix="max", gaussian_blend_lambda=0.5).to(runner.device)

    total_fgd, total_fsd, total_n = 0.0, 0.0, 0
    for batch in test_loader:
        images = batch["image"].to(runner.device)
        teacher_features = batch["teacher_features"].to(runner.device)
        image_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [*batch["image_id"]]

        # 前向得到学生特征（S16→对齐到教师尺度）
        feats = runner.backbone(images)
        s16 = feats[2]
        aligned = runner.feature_adapter(s16)
        if aligned.shape[-2:] != teacher_features.shape[-2:]:
            aligned = F.interpolate(aligned, size=teacher_features.shape[-2:], mode="bilinear", align_corners=False)

        # 构建前景/背景权重图到特征分辨率：优先NPZ（test_features_dir），缺失时JSON（若提供）
        Hf, Wf = aligned.shape[-2], aligned.shape[-1]
        fg_map, bg_map = None, None
        if load_weights_from_npz is not None:
            try:
                fg_map, bg_map = load_weights_from_npz(image_ids, str(test_features_dir), (Hf, Wf))
                fg_map = fg_map.to(runner.device)
                bg_map = bg_map.to(runner.device)
            except Exception:
                fg_map, bg_map = None, None
        if (fg_map is None or bg_map is None) and gt_json_dir is not None:
            fg_map, bg_map = build_fg_bg_from_ids(image_ids, str(gt_json_dir), (Hf, Wf))
            fg_map = fg_map.to(runner.device)
            bg_map = bg_map.to(runner.device)

        # 若仍不可用（既没有NPZ也没有JSON），则无法计算FGD/FSD，跳过该batch
        if fg_map is None or bg_map is None:
            continue

        # 分别计算FGD/FSD的loss（均值化）
        fgd_val = fgd_loss_fn(aligned, teacher_features, fg_map=fg_map, bg_map=bg_map, edge_map=None).item()
        fsd_val = fsd_loss_fn(aligned, teacher_features, fg_map=fg_map, bg_map=bg_map).item()

        bs = images.size(0)
        total_fgd += fgd_val * bs
        total_fsd += fsd_val * bs
        total_n += bs

    if total_n > 0:
        metrics_fgd = total_fgd / total_n
        metrics_fsd = total_fsd / total_n

    # 合并扩展指标
    if metrics_fgd is not None:
        metrics["fgd_loss"] = metrics_fgd
    if metrics_fsd is not None:
        metrics["fsd_loss"] = metrics_fsd

    # 返回runner对象用于可视化（如果启用）
    return metrics, runner


@torch.no_grad()
def load_runner_for_visualization(
    checkpoint_path: Path,
    test_features_dir: Path,
    test_images_dir: Path,
    batch_size: int = 4,
    max_images: int = None,
):
    """仅用于可视化：加载runner与数据集，不进行任何评估或指标计算。"""
    # 仅构建数据集（限制样本数用于可视化）
    test_dataset = NPZWithImageIdDataset(str(test_features_dir), str(test_images_dir), max_images=max_images, input_size=1024)

    # 构建runner（与训练一致的组件）
    config = {
        "backbone": "yolov8",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "edge_mask_kernel_size": 3,
        "use_pos_weight": False,
        "enable_edge_mask_progressive": False,
    }
    runner = DistillSingleTester(config)

    # 加载checkpoint权重
    ckpt = torch.load(str(checkpoint_path), map_location=runner.device)
    runner.backbone.load_state_dict(ckpt["backbone"])  # backbone
    runner.edge_head.load_state_dict(ckpt.get("edge_head", {}), strict=False)  # edge head

    # 加载静态适配器（与训练脚本保持一致）
    if "edge_adapter" in ckpt:
        runner.edge_adapter.load_state_dict(ckpt["edge_adapter"], strict=False)
    if "feature_adapter" in ckpt:
        runner.feature_adapter.load_state_dict(ckpt["feature_adapter"], strict=False)

    return runner, test_dataset


def visualize_comparison(results, test_dataset, output_dir: Path, num_samples: int):
    """
    并排对比模式：每个样本一张大图，多模型并排对比
    布局：(num_models + 1)行 x 5列
    行：GT + 各模型预测
    列：原图、边缘GT/预测、边缘叠加、边缘误差、特征对比
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_models = len(results)
    print(f"🖼️  Generating {num_samples} comparison visualizations ({num_models} models)...")
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[idx]
            image = sample['image'].unsqueeze(0)
            teacher_features = sample['teacher_features'].unsqueeze(0)
            edge_gt = sample['edge_256x256']
            image_id = sample['image_id']
            
            # 收集所有模型的预测结果
            model_predictions = {}
            for result in results:
                runner = result['runner']
                runner.backbone.eval()
                runner.edge_adapter.eval()
                runner.edge_head.eval()
                runner.feature_adapter.eval()
                
                # 移动到正确的设备
                image_t = image.to(runner.device)
                teacher_features_t = teacher_features.to(runner.device)
                
                # 前向传播
                features = runner.backbone(image_t)
                s4_features = features[0]
                s16_features = features[2]
                
                aligned_s4 = runner.edge_adapter(s4_features)
                edge_logits = runner.edge_head(aligned_s4)
                edge_pred = torch.sigmoid(edge_logits[0, 0]).cpu().numpy()
                
                aligned_features = runner.feature_adapter(s16_features)
                if aligned_features.shape[-2:] != teacher_features_t.shape[-2:]:
                    aligned_features = F.interpolate(aligned_features, size=teacher_features_t.shape[-2:], mode="bilinear", align_corners=False)
                p4_feat = aligned_features[0].cpu().numpy()
                p4_mean = p4_feat.mean(axis=0)
                
                model_predictions[result['name']] = {
                    'edge_pred': edge_pred,
                    'p4_mean': p4_mean,
                }
            
            # 准备图像
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            img_resized = F.interpolate(
                torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0),
                size=(256, 256), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0).numpy()
            
            edge_gt_np = edge_gt.numpy()
            teacher_feat = teacher_features[0].cpu().numpy()
            teacher_mean = teacher_feat.mean(axis=0)
            
            # 创建对比图：(num_models+1)行 x 5列（+1是因为GT行）
            fig = plt.figure(figsize=(25, 5 * (num_models + 1)))
            gs = GridSpec(num_models + 1, 5, figure=fig, hspace=0.3, wspace=0.3)
            
            # 第一行：GT作为参考
            row = 0
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(img_np)
            ax.set_title(f"Input Image\n(ID: {image_id})", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[row, 1])
            ax.imshow(edge_gt_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title("Edge GT", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[row, 2])
            ax.imshow(img_resized)
            ax.contour(edge_gt_np, levels=[0.5], colors='green', linewidths=2, alpha=0.8)
            ax.set_title("GT Overlay", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[row, 3])
            ax.text(0.5, 0.5, "Ground Truth\nReference", ha='center', va='center', 
                   fontsize=16, fontweight='bold')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[row, 4])
            im = ax.imshow(teacher_mean, cmap='viridis')
            ax.set_title(f"Teacher Feature\nmean={teacher_mean.mean():.3f}", fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')
            
            # 后续行：各模型的预测
            for row, result in enumerate(results, start=1):
                model_name = result['name']
                pred = model_predictions[model_name]
                edge_pred = pred['edge_pred']
                p4_mean = pred['p4_mean']
                
                # 原图（只在第一行显示，其他行空白或显示模型名）
                ax = fig.add_subplot(gs[row, 0])
                if row == 1:
                    ax.imshow(img_np)
                ax.text(0.5, 0.5, model_name, ha='center', va='center', 
                       fontsize=14, fontweight='bold')
                ax.axis('off')
                
                # 边缘预测
                ax = fig.add_subplot(gs[row, 1])
                ax.imshow(edge_pred, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"Edge Prediction\nmean={edge_pred.mean():.3f}", fontsize=11)
                ax.axis('off')
                
                # 边缘叠加
                ax = fig.add_subplot(gs[row, 2])
                ax.imshow(img_resized)
                ax.contour(edge_pred, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
                ax.set_title("Pred Overlay", fontsize=11)
                ax.axis('off')
                
                # 边缘误差
                ax = fig.add_subplot(gs[row, 3])
                edge_diff = np.abs(edge_pred - edge_gt_np)
                im = ax.imshow(edge_diff, cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"Edge Error\nMAE={edge_diff.mean():.3f}", fontsize=11)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
                
                # 特征对比
                ax = fig.add_subplot(gs[row, 4])
                feat_diff = np.abs(p4_mean - teacher_mean)
                im = ax.imshow(feat_diff, cmap='hot')
                ax.set_title(f"Feature Diff\nMAE={feat_diff.mean():.3f}", fontsize=11)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
            
            # 保存
            save_path = output_dir / f"{image_id}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ {idx+1}/{num_samples}: {save_path.name}")
    
    print(f"\n🎉 Comparison visualizations complete!")


def visualize_separate(results, test_dataset, output_dir: Path, num_samples: int):
    """
    分别保存模式：每个模型每个样本分别保存（复用原有可视化逻辑）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🖼️  Generating {num_samples} separate visualizations for {len(results)} models...")
    
    with torch.no_grad():
        for result in results:
            model_name = result['name']
            runner = result['runner']
            runner.backbone.eval()
            runner.edge_adapter.eval()
            runner.edge_head.eval()
            runner.feature_adapter.eval()
            
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            for idx in range(min(num_samples, len(test_dataset))):
                sample = test_dataset[idx]
                image = sample['image'].unsqueeze(0).to(runner.device)
                teacher_features = sample['teacher_features'].unsqueeze(0).to(runner.device)
                edge_gt = sample['edge_256x256']
                image_id = sample['image_id']
                
                # 前向传播
                features = runner.backbone(image)
                s4_features = features[0]
                s16_features = features[2]
                
                aligned_s4 = runner.edge_adapter(s4_features)
                edge_logits = runner.edge_head(aligned_s4)
                edge_pred = torch.sigmoid(edge_logits[0, 0]).cpu().numpy()
                
                aligned_features = runner.feature_adapter(s16_features)
                if aligned_features.shape[-2:] != teacher_features.shape[-2:]:
                    aligned_features = F.interpolate(aligned_features, size=teacher_features.shape[-2:], mode="bilinear", align_corners=False)
                p4_feat = aligned_features[0].cpu().numpy()
                p4_mean = p4_feat.mean(axis=0)
                p4_energy = np.sqrt((p4_feat ** 2).mean(axis=0))
                
                teacher_feat = teacher_features[0].cpu().numpy()
                teacher_mean = teacher_feat.mean(axis=0)
                
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                
                # 创建可视化（2行5列，与原有逻辑一致）
                fig = plt.figure(figsize=(20, 10))
                gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)
                
                # Row 1: Original, Edge GT, Edge Pred, Edge Overlay, Edge Error
                ax0 = fig.add_subplot(gs[0, 0])
                ax0.imshow(img_np)
                ax0.set_title(f"Input Image\n(ID: {image_id})", fontsize=10)
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
                img_resized = F.interpolate(
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
                ax5.set_title(f"Student P4 Mean\nmean={p4_mean.mean():.3f}", fontsize=10)
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
                n_show = min(16, p4_feat.shape[0])
                grid_size = 4
                channel_grid = np.zeros((grid_size * 16, grid_size * 16))
                for i in range(n_show):
                    row, col = i // grid_size, i % grid_size
                    ch_data = p4_feat[i]
                    ch_resized = F.interpolate(
                        torch.from_numpy(ch_data).unsqueeze(0).unsqueeze(0),
                        size=(16, 16), mode='bilinear', align_corners=False
                    )[0, 0].numpy()
                    channel_grid[row*16:(row+1)*16, col*16:(col+1)*16] = ch_resized
                im9 = ax9.imshow(channel_grid, cmap='gray')
                ax9.set_title(f"First {n_show} Channels", fontsize=10)
                ax9.axis('off')
                
                # 保存
                save_path = model_output_dir / f"{image_id}_{model_name}_visualization.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"  ✅ {model_name}: {num_samples} visualizations saved to {model_output_dir}")
    
    print(f"\n🎉 Separate visualizations complete!")


def visualize_panel_4x4(results, test_dataset, output_dir: Path, num_samples: int):
    """
    4x4 汇总面板（每个样本一张图）：
    第1行：四张特征均值（四个模型）
    第2行：四张特征差异 |student mean - teacher mean|
    第3行：四张边缘预测
    第4行：四张边缘误差 |pred - edge_gt|
    仅展示这四类图，其它不需要。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 固定列顺序
    desired_order = ["MSE_Baseline", "FGD_NoEdgeBoost", "FGD_EdgeBoost", "FSD_NoEdgeBoost"]
    name_to_result = {r['name']: r for r in results}
    model_results = [name_to_result[n] for n in desired_order if n in name_to_result]
    if len(model_results) < 4:
        # 回退：补齐缺失的按原顺序（仅当有缺失时）
        seen = set(r['name'] for r in model_results)
        for r in results:
            if r['name'] not in seen and len(model_results) < 4:
                model_results.append(r)

    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[idx]
            image = sample['image'].unsqueeze(0)
            teacher_features = sample['teacher_features'].unsqueeze(0)
            edge_gt = sample['edge_256x256'].numpy()
            image_id = sample['image_id']

            # 计算teacher mean一次，并设定统一色标范围（用教师分布做参考）
            teacher_feat_np = teacher_features[0].cpu().numpy()
            teacher_mean = teacher_feat_np.mean(axis=0)
            vmin_mean = float(np.percentile(teacher_mean, 1))
            vmax_mean = float(np.percentile(teacher_mean, 99))

            # 收集四个模型的：p4_mean, feat_signed_diff, edge_pred, edge_err
            collected = []
            for result in model_results:
                runner = result['runner']
                runner.backbone.eval(); runner.edge_adapter.eval(); runner.edge_head.eval(); runner.feature_adapter.eval()

                img_t = image.to(runner.device)
                tea_t = teacher_features.to(runner.device)
                feats = runner.backbone(img_t)
                s4 = feats[0]; s16 = feats[2]
                aligned_s4 = runner.edge_adapter(s4)
                edge_logits = runner.edge_head(aligned_s4)
                edge_pred = torch.sigmoid(edge_logits[0, 0]).detach().cpu().numpy()

                aligned = runner.feature_adapter(s16)
                if aligned.shape[-2:] != tea_t.shape[-2:]:
                    aligned = F.interpolate(aligned, size=tea_t.shape[-2:], mode="bilinear", align_corners=False)
                p4_feat = aligned[0].detach().cpu().numpy()
                p4_mean = p4_feat.mean(axis=0)
                # 签名差异（零中心显示更直观）：student_mean - teacher_mean
                feat_signed_diff = p4_mean - teacher_mean
                edge_err = np.abs(edge_pred - edge_gt)

                collected.append({
                    'name': result['name'],
                    'p4_mean': p4_mean,
                    'feat_signed_diff': feat_signed_diff,
                    'edge_pred': edge_pred,
                    'edge_err': edge_err,
                })

            # 绘制 5x4 面板（顶部参考行 + 四行对比）
            fig = plt.figure(figsize=(16, 20))
            gs = GridSpec(5, 4, figure=fig, hspace=0.15, wspace=0.15)

            # 顶部参考行：原图（跨两列）、Teacher NPZ Mean、NPZ Edge
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            ax = fig.add_subplot(gs[0, 0:2])
            ax.imshow(img_np)
            ax.set_title(f"Input Image (ID: {image_id})", fontsize=11, fontweight='bold')
            ax.axis('off')

            ax = fig.add_subplot(gs[0, 2])
            im = ax.imshow(teacher_mean, cmap='viridis', vmin=vmin_mean, vmax=vmax_mean)
            ax.set_title("Teacher NPZ Feature Mean", fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.035)

            ax = fig.add_subplot(gs[0, 3])
            ax.imshow(edge_gt, cmap='gray', vmin=0, vmax=1)
            ax.set_title("NPZ Edge (256x256)", fontsize=10)
            ax.axis('off')

            # 顶部列名标签
            for c, item in enumerate(collected[:4]):
                ax = fig.add_subplot(gs[1, c])
                ax.set_title(item['name'], fontsize=11, fontweight='bold', pad=6)
                ax.axis('off')
            # 行1：特征均值（实际内容放在单独一层，使标题不被覆盖）
            for c, item in enumerate(collected[:4]):
                ax = fig.add_subplot(gs[1, c])
                im = ax.imshow(item['p4_mean'], cmap='viridis', vmin=vmin_mean, vmax=vmax_mean)
                ax.set_title("P4 Mean", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.035)

            # 行2：特征差异（签名差异，零中心双极色图）
            diffs = np.stack([it['feat_signed_diff'] for it in collected[:4]], axis=0)
            diff_abs_max = float(np.max(np.abs(diffs))) + 1e-6
            for c, item in enumerate(collected[:4]):
                ax = fig.add_subplot(gs[2, c])
                im = ax.imshow(item['feat_signed_diff'], cmap='seismic', vmin=-diff_abs_max, vmax=diff_abs_max)
                ax.set_title("Mean − Teacher (signed)", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.035)

            # 行3：边缘预测
            for c, item in enumerate(collected[:4]):
                ax = fig.add_subplot(gs[3, c])
                ax.imshow(item['edge_pred'], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"Edge Pred", fontsize=10)
                ax.axis('off')

            # 行4：边缘误差
            for c, item in enumerate(collected[:4]):
                ax = fig.add_subplot(gs[4, c])
                im = ax.imshow(item['edge_err'], cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"Edge Error", fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.035)

            # 总标题+图注
            fig.suptitle(
                "Top Ref: Input | Teacher NPZ Mean | NPZ Edge | "
                "Row1: P4 Mean | Row2: Mean−Teacher (signed, zero-centered) | "
                "Row3: Edge Pred | Row4: Edge Error\n"
                "Columns: MSE_Baseline | FGD_NoEdgeBoost | FGD_EdgeBoost | FSD_NoEdgeBoost",
                fontsize=12, y=0.96
            )

            save_path = output_dir / f"{image_id}_panel4x4.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ {idx+1}/{num_samples}: {save_path.name}")

    print("\n🎉 Panel 4x4 visualizations complete!")

def main():
    parser = argparse.ArgumentParser(description="Unified Model Test - Fair evaluation with unified metrics")
    parser.add_argument("--test-features-dir", type=str, default="/home/team/zouzhiyuan/dataset/sa1b/test/extracted",
                       help="测试集NPZ特征目录（默认使用固定1000张测试集）")
    parser.add_argument("--test-images-dir", type=str, default="/home/team/zouzhiyuan/dataset/sa1b/test",
                       help="测试集图片目录（默认使用固定1000张测试集）")
    parser.add_argument("--max-images", type=int, default=None,
                       help="限制测试样本数量（例如50，从数据集中选择前N张）")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="多个模型checkpoint路径")
    parser.add_argument("--names", nargs="+", required=True, help="与checkpoint一一对应的名称")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gt-json-dir", type=str, default=None, help="可选：提供SA风格GT json目录以计算FGD/FSD指标")
    parser.add_argument("--output", type=str, default="outputs/unified_model_test.txt")
    
    # 可视化选项
    parser.add_argument("--visualize", action="store_true", help="启用可视化，生成多模型对比图")
    parser.add_argument("--num-vis-samples", type=int, default=10, help="可视化样本数量（默认10）")
    parser.add_argument("--vis-output-dir", type=str, default="outputs/visualizations",
                       help="可视化输出目录")
    parser.add_argument("--vis-mode", type=str, default="comparison",
                       choices=["comparison", "separate", "panel"],
                       help="comparison: 并排对比, separate: 分别保存, panel: 4x4汇总面板")
    parser.add_argument("--visualize-only", action="store_true",
                       help="仅进行可视化（加载模型并对指定数量样本生成P4特征与边缘图），不做任何指标评估")
    
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.names):
        raise ValueError("--checkpoints 与 --names 数量必须一致")

    test_features_dir = Path(args.test_features_dir)
    test_images_dir = Path(args.test_images_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Unified Model Test - Start")
    print("=" * 80)
    print(f"test_features_dir: {test_features_dir}")
    print(f"test_images_dir  : {test_images_dir}")
    if args.max_images:
        print(f"max_images      : {args.max_images} (限制样本数量)")
    print(f"models           : {len(args.checkpoints)}")
    if args.visualize:
        print(f"visualize        : ✅ Enabled ({args.num_vis_samples} samples, mode={args.vis_mode})")
    print()

    results = []
    if args.visualize_only:
        # 仅为可视化创建runner，不做指标评估
        print("Visualization-only mode: skip metrics, only generate P4 feature and edge visualizations.")
        # 为每个checkpoint构建runner并加载权重（不做任何评估）
        results = []
        for name, ckpt in zip(args.names, args.checkpoints):
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
                continue
            print(f"Preparing model for visualization: {name}\n  ckpt: {ckpt_path}")
            runner_obj, test_dataset = load_runner_for_visualization(
                checkpoint_path=ckpt_path,
                test_features_dir=test_features_dir,
                test_images_dir=test_images_dir,
                batch_size=args.batch_size,
                max_images=args.max_images,
            )
            results.append({"name": name, "runner": runner_obj})
        # 直接进入可视化
        if args.visualize:
            print("\n" + "=" * 80)
            print("Generating Visualizations (visualize-only mode)...")
            print("=" * 80)
            vis_output_dir = Path(args.vis_output_dir)
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            num_vis = min(args.num_vis_samples, len(test_dataset))
            if args.vis_mode == "comparison":
                visualize_comparison(results, test_dataset, vis_output_dir, num_vis)
            elif args.vis_mode == "separate":
                visualize_separate(results, test_dataset, vis_output_dir, num_vis)
            else:
                visualize_panel_4x4(results, test_dataset, vis_output_dir, num_vis)
            print(f"\n✅ Visualizations saved to: {vis_output_dir}")
        return
    else:
        for name, ckpt in zip(args.names, args.checkpoints):
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
                continue
            print(f"Evaluating: {name}\n  ckpt: {ckpt_path}")
            try:
                metrics_dict, runner_obj = evaluate_checkpoint(
                    checkpoint_path=ckpt_path,
                    test_features_dir=test_features_dir,
                    test_images_dir=test_images_dir,
                    batch_size=args.batch_size,
                    max_images=args.max_images,
                    gt_json_dir=Path(args.gt_json_dir) if args.gt_json_dir else None,
                )
                results.append({"name": name, **metrics_dict, "runner": runner_obj})
                print(f"  Feature MSE : {metrics_dict['mse']:.6f}")
                print(f"  Feature MAE : {metrics_dict['mae']:.6f}")
                print(f"  Cosine Sim  : {metrics_dict['cosine_sim']:.6f}")
                print(f"  Edge Loss   : {metrics_dict['edge_loss']:.6f}\n")
                if 'fgd_loss' in metrics_dict:
                    print(f"  FGD Loss    : {metrics_dict['fgd_loss']:.6f}")
                if 'fsd_loss' in metrics_dict:
                    print(f"  FSD Loss    : {metrics_dict['fsd_loss']:.6f}")
            except Exception as e:
                print(f"  [ERROR] evaluate failed: {e}\n")

    if not results:
        print("No valid results. Exit.")
        return

    # 输出表格（从results中提取metrics，排除runner对象）
    print("\n" + "=" * 80)
    print("Unified Metrics Comparison")
    print("=" * 80)
    print(f"{'Model':<28} | {'Feat MSE':>10} | {'Feat MAE':>10} | {'Cosine Sim':>11} | {'Edge Loss':>10} | {'FGD Loss':>9} | {'FSD Loss':>9}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<28} | "
            f"{r['mse']:>10.6f} | "
            f"{r['mae']:>10.6f} | "
            f"{r['cosine_sim']:>11.6f} | "
            f"{r['edge_loss']:>10.6f} | "
            f"{r.get('fgd_loss', float('nan')):>9.6f} | "
            f"{r.get('fsd_loss', float('nan')):>9.6f}"
        )

    # 保存到文件（只保存metrics，不包含runner对象）
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Unified Metrics Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<28} | {'Feat MSE':>10} | {'Feat MAE':>10} | {'Cosine Sim':>11} | {'Edge Loss':>10} | {'FGD Loss':>9} | {'FSD Loss':>9}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(
                f"{r['name']:<28} | "
                f"{r['mse']:>10.6f} | "
                f"{r['mae']:>10.6f} | "
                f"{r['cosine_sim']:>11.6f} | "
                f"{r['edge_loss']:>10.6f} | "
                f"{r.get('fgd_loss', float('nan')):>9.6f} | "
                f"{r.get('fsd_loss', float('nan')):>9.6f}\n"
            )
    print(f"\nSaved to: {out_path}")
    
    # 可视化（如果启用）
    if args.visualize:
        print("\n" + "=" * 80)
        print("Generating Visualizations...")
        print("=" * 80)
        
        # 加载测试数据集
        test_dataset = NPZWithImageIdDataset(str(test_features_dir), str(test_images_dir), max_images=args.max_images, input_size=1024)
        num_vis = min(args.num_vis_samples, len(test_dataset))
        
        vis_output_dir = Path(args.vis_output_dir)
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.vis_mode == "comparison":
            visualize_comparison(results, test_dataset, vis_output_dir, num_vis)
        elif args.vis_mode == "separate":
            visualize_separate(results, test_dataset, vis_output_dir, num_vis)
        else:
            visualize_panel_4x4(results, test_dataset, vis_output_dir, num_vis)
        
        print(f"\n✅ Visualizations saved to: {vis_output_dir}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()



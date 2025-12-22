#!/usr/bin/env python3
"""
独立统一测试脚本：在固定测试集上，使用统一指标（Feature MSE/MAE、Cosine、Edge BCE+Dice）
对已训练好的模型进行公平对比。

严格复用当前训练脚本的关键逻辑：
- 复用数据集类 NPZWithImageIdDataset（真实图片+NPZ严格配对）
- 复用模型组件创建方式（YOLOv8Backbone、UniversalEdgeHead、SimpleAdapter）
- 复用统一指标评估实现 validate_unified_metrics 的核心计算流程

用法示例：
python VFMKD/tools/unified_model_test.py \
  --features-dir VFMKD/outputs/features_v1_300 \
  --images-dir datasets/An_.../An_... \
  --checkpoints \
    VFMKD/outputs/testFGDFSD/mse_no_edge_boost_vis/best_model.pth \
    VFMKD/outputs/testFGDFSD/fgd_no_edge_boost_vis/best_model.pth \
    VFMKD/outputs/testFGDFSD/fgd_edge_boost_vis/best_model.pth \
  --names MSE FGD FGD+Edge \
  --batch-size 4 \
  --output VFMKD/outputs/testFGDFSD/unified_model_test.txt
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

# 路径：将项目根目录加入sys.path，保持与训练脚本一致
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 仅导入我们需要复用的类与方法（保持一致性）
from VFMKD.tools.core.train_distill_single_test import (
    NPZWithImageIdDataset,
    DistillSingleTester,
)


def create_fixed_test_subset(dataset: torch.utils.data.Dataset, test_ratio: float = 0.1, seed: int = 42) -> Subset:
    """与训练脚本一致的固定测试划分：使用固定seed的随机打乱，取最后10%为测试集。"""
    torch.manual_seed(seed)
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    test_size = max(1, int(n * test_ratio))
    test_indices = indices[-test_size:]
    return Subset(dataset, test_indices)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, features_dir: Path, images_dir: Path, batch_size: int = 4,
                        test_ratio: float = 0.1, seed: int = 42) -> dict:
    """
    加载checkpoint，构建与训练一致的模型组件，然后在固定测试集上用统一指标评估。
    返回：{"mse":..., "mae":..., "cosine_sim":..., "edge_loss":...}
    """
    # 数据集（严格复用）
    dataset = NPZWithImageIdDataset(str(features_dir), str(images_dir), max_images=None, input_size=1024)
    test_dataset = create_fixed_test_subset(dataset, test_ratio=test_ratio, seed=seed)
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

    # 加载checkpoint（严格对应键名）
    ckpt = torch.load(str(checkpoint_path), map_location=runner.device)
    runner.backbone.load_state_dict(ckpt["backbone"])  # 严格
    # edge_head含动态channel_aligners，加载用strict=False与现有可视化一致
    runner.edge_head.load_state_dict(ckpt["edge_head"], strict=False)
    runner.feature_adapter.load_state_dict(ckpt["feature_adapter"])  # 严格

    # 使用训练脚本内的统一指标实现，保持计算细节一致
    metrics = runner.validate_unified_metrics(test_loader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified Model Test - Fair evaluation with unified metrics")
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="多个模型checkpoint路径")
    parser.add_argument("--names", nargs="+", required=True, help="与checkpoint一一对应的名称")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="VFMKD/outputs/testFGDFSD/unified_model_test.txt")
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.names):
        raise ValueError("--checkpoints 与 --names 数量必须一致")

    features_dir = Path(args.features_dir)
    images_dir = Path(args.images_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Unified Model Test - Start")
    print("=" * 80)
    print(f"features_dir: {features_dir}")
    print(f"images_dir  : {images_dir}")
    print(f"models      : {len(args.checkpoints)}")
    print()

    results = []
    for name, ckpt in zip(args.names, args.checkpoints):
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
            continue
        print(f"Evaluating: {name}\n  ckpt: {ckpt_path}")
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                features_dir=features_dir,
                images_dir=images_dir,
                batch_size=args.batch_size,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            results.append({"name": name, **metrics})
            print(f"  Feature MSE : {metrics['mse']:.6f}")
            print(f"  Feature MAE : {metrics['mae']:.6f}")
            print(f"  Cosine Sim  : {metrics['cosine_sim']:.6f}")
            print(f"  Edge Loss   : {metrics['edge_loss']:.6f}\n")
        except Exception as e:
            print(f"  [ERROR] evaluate failed: {e}\n")

    if not results:
        print("No valid results. Exit.")
        return

    # 输出表格
    print("\n" + "=" * 80)
    print("Unified Metrics Comparison")
    print("=" * 80)
    print(f"{'Model':<28} | {'Feat MSE':>10} | {'Feat MAE':>10} | {'Cosine Sim':>11} | {'Edge Loss':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<28} | "
            f"{r['mse']:>10.6f} | "
            f"{r['mae']:>10.6f} | "
            f"{r['cosine_sim']:>11.6f} | "
            f"{r['edge_loss']:>10.6f}"
        )

    # 保存到文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Unified Metrics Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<28} | {'Feat MSE':>10} | {'Feat MAE':>10} | {'Cosine Sim':>11} | {'Edge Loss':>10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(
                f"{r['name']:<28} | "
                f"{r['mse']:>10.6f} | "
                f"{r['mae']:>10.6f} | "
                f"{r['cosine_sim']:>11.6f} | "
                f"{r['edge_loss']:>10.6f}\n"
            )
    print(f"\nSaved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""Compare multiple bbox selection strategies on the same images and render side-by-side panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tools.core.bbox.test_bbox_strategies import (
    apply_strategy,
    build_candidate_visuals,
    PALETTE_24,
    prepare_edge_map,
)


def auto_select_images(data_dir: Path,
                       num_images: int,
                       min_area_ratio: float = 8e-4,
                       max_area_ratio: float = 5e-2,
                       min_pred_iou: float = 0.92,
                       min_stability: float = 0.97) -> List[str]:
    candidates = []
    json_files = sorted(data_dir.glob('sa_*.json'))
    for jp in tqdm(json_files, desc="Scanning JSON", leave=False):
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        H = data['image']['height']
        W = data['image']['width']
        total_area = float(H * W)

        good = []
        for ann in data.get('annotations', []):
            area = float(ann.get('area', 0))
            if area <= 0:
                continue
            ratio = area / total_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            if ann.get('predicted_iou', 0.0) < min_pred_iou:
                continue
            if ann.get('stability_score', 0.0) < min_stability:
                continue
            good.append((ratio, ann.get('predicted_iou', 0.0)))

        if not good:
            continue
        count = len(good)
        avg_iou = float(np.mean([g[1] for g in good]))
        sum_ratio = float(np.sum([g[0] for g in good]))
        candidates.append((count, avg_iou, sum_ratio, jp.stem))

    if not candidates:
        raise RuntimeError("No candidates found for auto selection")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    stems = [c[3] for c in candidates[:num_images]]
    if len(stems) < num_images:
        extra = [c[3] for c in candidates[len(stems):]]
        stems.extend(extra[: max(0, num_images - len(stems))])
    return stems[:num_images]


def draw_boxes(ax, image_bgr: np.ndarray, candidates: List[Dict[str, Any]], title: str) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    for idx, cand in enumerate(candidates):
        box = cand.get('box')
        if not box:
            continue
        color = PALETTE_24[idx % len(PALETTE_24)]
        lw = 3 if cand.get('selected') else 1.5
        rect = plt.Rectangle((box[0], box[1]), box[2], box[3], linewidth=lw,
                             edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = f"#{cand['rank']} ({cand['score']:.2f})"
        ax.text(box[0], max(5, box[1] - 5), label, color=color, fontsize=11,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def create_overlay(image_bgr: np.ndarray,
                   orig_masks: List[np.ndarray],
                   palette: List[tuple],
                   alpha: float = 0.45) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = image_rgb.copy()
    for idx, mask in enumerate(orig_masks):
        if mask is None or mask.size == 0:
            continue
        color = np.array(palette[idx % len(palette)], dtype=np.float32)
        mask_bool = mask.astype(bool)
        if mask_bool.sum() == 0:
            continue
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def render_panel(image_bgr: np.ndarray,
                 strategies: List[str],
                 strategy_boxes: Dict[str, Dict[str, Any]],
                 save_path: Path) -> None:
    rows = len(strategies) * 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(32, 7 * len(strategies)), constrained_layout=True)

    for s_idx, strat in enumerate(strategies):
        info = strategy_boxes.get(strat, {})
        candidates = info.get('candidates', [])
        overlay_img, mosaic_img, score_columns, display_candidates, mosaic_meta, weight_formula = build_candidate_visuals(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), candidates, max_display=10)

        row_visual = s_idx * 2
        row_scores = row_visual + 1

        ax_boxes = axes[row_visual, 0]
        draw_boxes(ax_boxes, image_bgr, candidates[:10], f"{strat} - Boxes")

        ax_masks = axes[row_visual, 1]
        ax_masks.imshow(mosaic_img)
        ax_masks.set_title('Top Masks', fontsize=14, fontweight='bold')
        ax_masks.axis('off')
        cell_size, cols_used = mosaic_meta
        if display_candidates:
            cell_w = cell_size
            for idx, cand in enumerate(display_candidates):
                row_idx = idx // cols_used
                col_idx = idx % cols_used
                x = col_idx * cell_w
                y = row_idx * cell_size
                ax_masks.text(x + 12, y + 22, f"#{cand['rank']} ({cand['score']:.2f})",
                               color='white', fontsize=11, weight='bold')
                if cand.get('selected'):
                    rect = plt.Rectangle((x + 1, y + 1), cell_w - 2, cell_size - 2,
                                         linewidth=2, edgecolor='white', facecolor='none')
                    ax_masks.add_patch(rect)

        ax_overlay = axes[row_visual, 2]
        ax_overlay.imshow(overlay_img)
        ax_overlay.set_title('Overlay', fontsize=14, fontweight='bold')
        ax_overlay.axis('off')

        ax_formula = axes[row_visual, 3]
        ax_formula.axis('off')
        if weight_formula:
            ax_formula.set_title('Weight Formula', fontsize=14, fontweight='bold')
            ax_formula.text(0.02, 0.5, weight_formula, fontsize=12, va='center', ha='left', wrap=True)

        axes[row_visual, 4].axis('off')

        for col in range(cols):
            ax = axes[row_scores, col]
            ax.axis('off')
            if col < len(score_columns):
                if col == 0:
                    ax.set_title('Scores', fontsize=13, fontweight='bold')
                y = 0.96
                for block in score_columns[col]:
                    if not block:
                        continue
                    headline = block[0]
                    rank_str = headline.split()[0]
                    try:
                        rank_num = int(rank_str.replace('#', '').replace(':', ''))
                    except Exception:
                        rank_num = 1
                    color = PALETTE_24[(rank_num - 1) % len(PALETTE_24)]
                    ax.text(0.02, y, '■', color=color, fontsize=9.5, va='top')
                    ax.text(0.04, y, headline, fontsize=9.5, va='top', ha='left', wrap=True)
                    y -= 0.12
                    for line in block[1:]:
                        ax.text(0.04, y, line, fontsize=8.5, va='top', ha='left', wrap=True)
                        y -= 0.1
                    y -= 0.045

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple bbox extraction strategies side-by-side')
    parser.add_argument('--data-dir', type=str, required=True, help='SA-1B directory containing images and JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for comparison panels')
    parser.add_argument('--strategies', type=str, default='A,D', help='Comma-separated strategies (must be subset of A,D)')
    parser.add_argument('--selection-file', type=str, default='', help='Optional file listing image stems to visualize')
    parser.add_argument('--num-images', type=int, default=50, help='Number of images to visualize when auto-selecting')
    parser.add_argument('--nms-iou-threshold', type=float, default=0.4, help='NMS IoU threshold for strategies that require it')
    parser.add_argument('--mask-size', type=int, default=256, help='Mask resize dimension')
    parser.add_argument('--max-channels', type=int, default=1, help='Maximum number of instances to display per strategy')
    parser.add_argument('--no-visualize', action='store_true', help='Skip rendering panels (only run strategies)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_strategies = [s.strip().upper() for s in args.strategies.split(',') if s.strip()]
    allowed = ['A', 'D']
    strategies = []
    for s in raw_strategies:
        if s not in allowed:
            raise ValueError(f'Unsupported strategy {s}; only A and D are allowed')
        if s not in strategies:
            strategies.append(s)
    if not strategies:
        strategies = ['A', 'D']

    if args.selection_file:
        selection_path = Path(args.selection_file)
        stems = [line.strip() for line in selection_path.read_text().splitlines() if line.strip()]
        stems = stems[:args.num_images]
    else:
        stems = auto_select_images(data_dir, args.num_images)
        (output_dir / 'selected_images.txt').write_text('\n'.join(stems), encoding='utf-8')

    for stem in tqdm(stems, desc='Rendering panels'):
        json_path = data_dir / f'{stem}.json'
        image_path = data_dir / f'{stem}.jpg'
        if not json_path.exists() or not image_path.exists():
            print(f'⚠️ Missing json/image for {stem}, skipping')
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'⚠️ Cannot read image {image_path}, skipping')
            continue

        strategy_boxes: Dict[str, Dict[str, Any]] = {}
        for strat in strategies:
            edge_map = prepare_edge_map(image)
            boxes, masks, extra = apply_strategy(
                strat,
                data,
                str(json_path),
                image,
                edge_map,
                args.nms_iou_threshold,
                mask_size=args.mask_size,
                max_channels=args.max_channels,
            )
            strategy_boxes[strat] = {
                'boxes': boxes,
                'masks': masks,
                'scores': extra.get('scores'),
                'orig_masks': extra.get('orig_masks'),
                'ranking': extra.get('ranking'),
                'candidates': extra.get('candidates', []),
            }

        if not args.no_visualize:
            panel_path = output_dir / f'{stem}_comparison.png'
            render_panel(image, strategies, strategy_boxes, panel_path)


if __name__ == '__main__':
    main()

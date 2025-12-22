#!/usr/bin/env python3
"""
Panel-A 批量评测CLI：
- 默认设备 cuda:6
- 默认从 test 相关目录（--images-dir）取前1000张图片运行，
  仅可视化10张（均匀抽样），其余只统计不出图。
- 输出：选中的10张面板PNG + 全量CSV统计。
"""

import argparse
from pathlib import Path
import csv

from tools.core.prompt.panel_runner import run_panel


def list_images(root: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [p for p in sorted(root.rglob('*')) if p.suffix.lower() in exts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', type=str, required=True)
    ap.add_argument('--student-ckpt', type=str, required=True)
    ap.add_argument('--teacher-npz-dir', type=str, required=True, help='与图片同名或可匹配的NPZ所在目录')
    ap.add_argument('--sam2-cfg', type=str, default='sam2.1/sam2.1_hiera_b+.yaml')
    ap.add_argument('--sam2-ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda:6')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--visualize', type=int, default=10)
    ap.add_argument('--out-dir', type=str, default=None)
    args = ap.parse_args()

    images_root = Path(args.images_dir)
    out_root = Path(args.out_dir) if args.out_dir else (images_root / 'panel_a_outputs')
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_root)[: args.limit]
    vis_indices = set()
    if len(imgs) > 0 and args.visualize > 0:
        step = max(1, len(imgs) // args.visualize)
        vis_indices = set(range(0, len(imgs), step))

    csv_path = out_root / 'stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'image',
            'cos_student_npz','mae_student_npz','l2_student_npz',
            'cos_student_teacher','mae_student_teacher','l2_student_teacher',
            'cos_npz_teacher','mae_npz_teacher','l2_npz_teacher',
            'iou_student_npz','iou_student_teacher','iou_npz_teacher',
            'panel_path'
        ]
        writer.writerow(header)

        for idx, img in enumerate(imgs):
            # 简单的NPZ配对规则：同名不同后缀、或同基名在teacher-npz-dir下的 .npz
            base = img.stem
            npz_path = Path(args.teacher_npz_dir) / f'{base}_features.npz'
            if not npz_path.exists():
                npz_path = Path(args.teacher_npz_dir) / f'{base}.npz'
            if not npz_path.exists():
                # 跳过无NPZ的样本
                continue

            panel_out = None
            if idx in vis_indices:
                panel_out = out_root / f'{base}_panel.png'
            stats = run_panel(
                image_path=img,
                student_weights=Path(args.student_ckpt),
                teacher_npz_path=npz_path,
                sam2_cfg=args.sam2_cfg,
                sam2_ckpt=Path(args.sam2_ckpt) if args.sam2_ckpt else None,
                device=args.device,
                output_path=panel_out,
            )
            writer.writerow([
                str(img),
                stats.get('cos_student_npz'), stats.get('mae_student_npz'), stats.get('l2_student_npz'),
                stats.get('cos_student_teacher'), stats.get('mae_student_teacher'), stats.get('l2_student_teacher'),
                stats.get('cos_npz_teacher'), stats.get('mae_npz_teacher'), stats.get('l2_npz_teacher'),
                stats.get('iou_student_npz'), stats.get('iou_student_teacher'), stats.get('iou_npz_teacher'),
                stats.get('output') if panel_out else '',
            ])

    print('[OK] 批量完成，统计已保存：', str(csv_path))


if __name__ == '__main__':
    main()



#!/usr/bin/env python3
"""
Panel-A 命令行入口。
示例：
python tools/core/prompt/panel_cli.py \
  --image /path/to/img.jpg \
  --student-ckpt /path/to/student_adapter.pt \
  --teacher-npz /path/to/teacher_p4.npz \
  --sam2-ckpt /home/team/zouzhiyuan/vfmkd/weights/sam2.1_hiera_base_plus.pt \
  --out /home/team/zouzhiyuan/vfmkd/outputs/panel_a.png
"""

import argparse
from pathlib import Path
from tools.core.prompt.panel_runner import run_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--student-ckpt', type=str, required=True, help='包含adapter(必需)与可选backbone的权重文件')
    ap.add_argument('--teacher-npz', type=str, required=True)
    ap.add_argument('--sam2-cfg', type=str, default='sam2.1/sam2.1_hiera_b+.yaml')
    ap.add_argument('--sam2-ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda:6')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    stats = run_panel(
        image_path=Path(args.image),
        student_weights=Path(args.student_ckpt),
        teacher_npz_path=Path(args.teacher_npz),
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=Path(args.sam2_ckpt) if args.sam2_ckpt else None,
        device=args.device,
        output_path=Path(args.out) if args.out else None,
    )
    print('[OK] Panel saved:', stats.get('output'))
    for k, v in stats.items():
        if k != 'output':
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()



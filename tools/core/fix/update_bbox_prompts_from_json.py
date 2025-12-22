#!/usr/bin/env python3
"""批量根据 SA-1B JSON 生成实例框/掩码，并原子写回 NPZ。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 JSON 更新 NPZ 中的 box prompts")
    parser.add_argument('--npz-dir', type=str, required=True, help='NPZ 根目录（如 /home/team/zouzhiyuan/dataset/sa1b/extracted）')
    parser.add_argument('--json-dir', type=str, required=True, help='SA-1B JSON 目录（与 JPG 同级）')
    parser.add_argument('--log', type=str, required=True, help='日志输出路径')
    parser.add_argument('--max-files', type=int, default=0, help='限制处理文件数，0 表示不限')
    parser.add_argument('--skip-if-flag', action='store_true', help='若 NPZ 已存在 box_flag=1，则跳过')
    parser.add_argument('--strategy-tag', type=str, default='strategy_A', help='写入 box_prompts_meta 的策略名称')
    parser.add_argument('--cuda-device-id', type=int, default=4, help='CUDA 设备 ID（默认 4，遵循禁用 0/3 的策略）')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU 模式')
    parser.add_argument('--nms-iou', type=float, default=0.5, help='实例提取器 NMS IoU 阈值')
    parser.add_argument('--score-threshold', type=float, default=0.8, help='实例提取器动态 K 分数阈值')
    parser.add_argument('--max-instances', type=int, default=5, help='最多保留的实例数')
    parser.add_argument('--dry-run', action='store_true', help='仅统计将要处理的文件，不写回')
    return parser.parse_args()


def collect_npz_files(root: Path) -> List[Path]:
    patterns = ('*_features.npz', '*_sam2_features.npz')
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(sorted(root.rglob(pat)))

    seen = set()
    unique: List[Path] = []
    for npz_path in candidates:
        stem = npz_path.name
        key = stem.replace('_sam2_features.npz', '').replace('_features.npz', '')
        if key in seen:
            continue
        seen.add(key)
        unique.append(npz_path)
    return unique


def ensure_sys_path() -> None:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def build_extractor(args: argparse.Namespace):  # pragma: no cover - 简单封装
    ensure_sys_path()
    from tools.core.bbox.sa1b_bbox_extractor import SA1BInstanceBoxExtractor

    use_cuda = (not args.cpu)
    extractor = SA1BInstanceBoxExtractor(
        nms_iou_threshold=args.nms_iou,
        use_cuda=use_cuda,
        cuda_device_id=args.cuda_device_id,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
    )
    return extractor


def to_xyxy(boxes_xywh: List[List[float]]) -> np.ndarray:
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(boxes_xywh, dtype=np.float32)
    arr_xyxy = arr.copy()
    arr_xyxy[:, 2] = arr[:, 0] + arr[:, 2]
    arr_xyxy[:, 3] = arr[:, 1] + arr[:, 3]
    return arr_xyxy


def atomic_write_npz(target_path: Path, payload: Dict[str, np.ndarray]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix='.npz', dir=str(target_path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **payload)
        os.replace(tmp_name, target_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def resolve_json(json_dir: Path, stem: str) -> Path:
    return json_dir / f"{stem}.json"


def main() -> None:  # pragma: no cover - CLI 脚本入口
    args = parse_args()
    npz_root = Path(args.npz_dir)
    json_root = Path(args.json_dir)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = collect_npz_files(npz_root)
    if args.max_files and args.max_files > 0:
        npz_files = npz_files[:args.max_files]

    extractor = build_extractor(args)

    processed = 0
    skipped = 0
    missing = 0
    errors = 0

    with open(log_path, 'w', encoding='utf-8') as log_f:
        for idx, npz_path in enumerate(npz_files, 1):
            stem = npz_path.stem.replace('_features', '').replace('_sam2_features', '')
            json_path = resolve_json(json_root, stem)
            if not json_path.exists():
                log_f.write(f'[SKIP] json not found -> {stem}\n')
                log_f.flush()
                missing += 1
                continue

            try:
                data = load_npz(npz_path)
            except Exception as exc:  # pragma: no cover - I/O safeguard
                log_f.write(f'[SKIP] failed to read {npz_path}: {exc}\n')
                log_f.flush()
                skipped += 1
                continue

            box_flag = data.get('box_flag', np.array(0, dtype=np.uint8))
            box_flag_val = int(box_flag.item()) if isinstance(box_flag, np.ndarray) else int(box_flag)
            if args.skip_if_flag and box_flag_val == 1:
                log_f.write(f'[SKIP] already processed -> {stem}\n')
                log_f.flush()
                skipped += 1
                continue

            try:
                result = extractor.extract_top_boxes_simple(str(json_path))
            except Exception as exc:  # pragma: no cover - extractor safeguard
                log_f.write(f'[ERR] extractor failed {stem}: {exc}\n')
                log_f.flush()
                errors += 1
                continue

            boxes_xywh = result.get('boxes', []) or []
            count = result.get('total_available', len(boxes_xywh))
            if count != len(boxes_xywh):
                count = len(boxes_xywh)

            boxes_xyxy = to_xyxy(boxes_xywh)
            masks = result.get('masks', np.zeros((extractor.max_instances, extractor.mask_size, extractor.mask_size), dtype=np.uint8))
            masks = np.asarray(masks)
            masks = masks[:count].astype(np.uint8, copy=False)

            payload = dict(data)  # 拷贝原有键
            payload['box_prompts_xyxy'] = boxes_xyxy.astype(np.float32)
            payload['box_prompts_count'] = np.array(count, dtype=np.int32)
            payload['box_prompts_masks_256'] = masks  # K×256×256
            payload['box_prompts_meta'] = np.array([args.strategy_tag], dtype=object)
            payload['box_flag'] = np.array(1, dtype=np.uint8)

            if args.dry_run:
                log_f.write(f'[DRY] {idx}/{len(npz_files)} {stem} count={count}\n')
                log_f.flush()
                processed += 1
                continue

            try:
                atomic_write_npz(npz_path, payload)
            except Exception as exc:  # pragma: no cover - write safeguard
                log_f.write(f'[ERR] write failed {stem}: {exc}\n')
                log_f.flush()
                errors += 1
                continue

            log_f.write(f'[OK] {idx}/{len(npz_files)} {stem} count={count}\n')
            log_f.flush()
            processed += 1

    summary = (
        f'[DONE] processed={processed} skipped={skipped} missing_json={missing} '
        f'errors={errors} total={len(npz_files)} dry_run={args.dry_run}'
    )
    print(summary)


if __name__ == '__main__':  # pragma: no cover - 脚本执行入口
    main()



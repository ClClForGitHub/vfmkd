#!/usr/bin/env python3
"""
独立的 geometry_color 策略应用脚本

功能：
- 扫描已存在的NPZ文件（*_features.npz 或 *_sam2_features.npz）
- 从对应的JSON文件读取标注
- 应用geometry_color策略选择框和掩码
- 原子写回NPZ文件（参考tools/core/fix的实现）
- 支持flag标记和防重复机制

输出NPZ格式：
- has_bbox: bool, 是否有框
- num_bboxes: int32, 框的数量
- bboxes: float32 array (N, 4), [x, y, w, h]格式
- masks: object array (N,), 每个元素是(H, W) uint8掩码数组（仅当has_bbox=True时存在）
- geometry_color_flag: uint8, 标记是否已处理（1=已处理）

使用方法：
    python tools/core/bbox/apply_geometry_color_strategy.py \\
        --npz-dir /path/to/npz \\
        --json-dir /path/to/json \\
        --log /path/to/log.txt \\
        [--max-instances 1] \\
        [--max-files 100] \\
        [--skip-if-processed]
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# 添加项目路径
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入必要的函数（从原脚本）
from pycocotools import mask as mask_utils
from tools.core.bbox.test_bbox_strategies import (
    load_sa_json,
    rle_to_binary,
    compute_strategy_geometry_color,
)


def atomic_write_npz(target_path: Path, payload: Dict[str, Any]) -> None:
    """
    原子写回NPZ文件（参考 tools/core/fix 的实现）
    """
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


def apply_geometry_color_strategy(
    image_path: Path,
    json_path: Path,
    max_instances: int = 1
) -> Dict[str, Any]:
    """
    应用 geometry_color 策略，选择框和掩码
    
    Args:
        image_path: 图片路径
        json_path: JSON路径
        max_instances: 最多选择的实例数
    
    Returns:
        {
            'has_bbox': bool,  # 是否有框
            'bboxes': np.ndarray,  # shape: (N, 4) [x, y, w, h]
            'masks': List[np.ndarray],  # N个掩码数组
        }
    """
    # 1. 读取图片和JSON
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    H, W = image_bgr.shape[:2]
    
    sa_data = load_sa_json(str(json_path))
    annotations = sa_data.get('annotations', [])
    
    if len(annotations) == 0:
        return {
            'has_bbox': False,
            'bboxes': np.empty((0, 4), dtype=np.float32),
            'masks': [],
        }
    
    # 2. 准备data字典（包含图像信息）
    data = {
        'image': {
            'height': H,
            'width': W,
            'h': H,
            'w': W,
        }
    }
    
    # 3. 应用策略选择框和掩码
    selected_components = compute_strategy_geometry_color(
        data=data,
        annotations=annotations,
        image_rgb=image_bgr,  # BGR格式
        clip_data=None,  # 不使用CLIP
        max_instances=max_instances,
        max_display=10,
        debug_trace=None,  # 不需要可视化，不保存trace
    )
    
    # 3. 提取框和掩码
    if len(selected_components) == 0:
        return {
            'has_bbox': False,
            'bboxes': np.empty((0, 4), dtype=np.float32),
            'masks': [],
        }
    
    bboxes = []
    masks = []
    for comp in selected_components:
        bbox = comp['box']  # [x, y, w, h]
        mask = comp['mask']  # [H, W] uint8
        bboxes.append(bbox)
        masks.append(mask)
    
    return {
        'has_bbox': True,
        'bboxes': np.array(bboxes, dtype=np.float32),  # (N, 4)
        'masks': masks,  # List of (H, W) arrays
    }


def collect_npz_files(root: Path) -> List[Path]:
    """收集所有NPZ文件（支持两种命名模式），并去重（参考fix目录的通用实现）"""
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


def load_npz(npz_path: Path) -> Dict[str, Any]:
    """加载NPZ文件（参考fix目录的通用实现）"""
    with np.load(str(npz_path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def resolve_json(json_dir: Path, stem: str) -> Path:
    """解析JSON文件路径（参考fix目录的通用实现）"""
    return json_dir / f"{stem}.json"


def resolve_image(image_dir: Path, stem: str) -> Optional[Path]:
    """解析图片文件路径（尝试多种扩展名）"""
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_single_file(args_tuple: Tuple[Path, Path, Path, int, bool]) -> Dict[str, Any]:
    """
    处理单个文件的worker函数（用于多进程）
    
    Args:
        args_tuple: (npz_path, json_root, image_root, max_instances, skip_if_flag)
    
    Returns:
        {
            'stem': str,
            'status': 'ok' | 'skip' | 'error',
            'message': str,
            'bboxes_count': int
        }
    """
    npz_path, json_root, image_root, max_instances, skip_if_flag = args_tuple
    
    stem = npz_path.stem.replace('_features', '').replace('_sam2_features', '')
    json_path = resolve_json(json_root, stem)
    image_path = resolve_image(image_root, stem)
    
    # 检查文件是否存在
    if not json_path.exists():
        return {'stem': stem, 'status': 'skip', 'message': f'JSON not found', 'bboxes_count': 0}
    
    if image_path is None or not image_path.exists():
        return {'stem': stem, 'status': 'skip', 'message': f'Image not found', 'bboxes_count': 0}
    
    # 加载NPZ
    try:
        data = load_npz(npz_path)
    except Exception as exc:
        return {'stem': stem, 'status': 'error', 'message': f'Failed to read NPZ: {exc}', 'bboxes_count': 0}
    
    # 检查flag
    if skip_if_flag:
        geometry_color_flag = data.get('geometry_color_flag', np.array(0, dtype=np.uint8))
        flag_val = int(geometry_color_flag.item()) if isinstance(geometry_color_flag, np.ndarray) else int(geometry_color_flag)
        if flag_val == 1:
            return {'stem': stem, 'status': 'skip', 'message': 'already processed', 'bboxes_count': 0}
    
    # 应用策略
    try:
        result = apply_geometry_color_strategy(
            image_path=image_path,
            json_path=json_path,
            max_instances=max_instances
        )
    except Exception as exc:
        return {'stem': stem, 'status': 'error', 'message': f'Strategy failed: {exc}', 'bboxes_count': 0}
    
    # 更新NPZ数据
    payload = dict(data)
    payload['has_bbox'] = np.array(result['has_bbox'], dtype=bool)
    payload['num_bboxes'] = np.array(len(result['bboxes']), dtype=np.int32)
    payload['bboxes'] = result['bboxes']
    
    if result['has_bbox'] and len(result['masks']) > 0:
        payload['masks'] = np.array(result['masks'], dtype=object)
    
    payload['geometry_color_flag'] = np.array(1, dtype=np.uint8)
    
    # 原子写回
    try:
        atomic_write_npz(npz_path, payload)
        return {
            'stem': stem,
            'status': 'ok',
            'message': 'success',
            'bboxes_count': len(result['bboxes'])
        }
    except Exception as exc:
        return {'stem': stem, 'status': 'error', 'message': f'Write failed: {exc}', 'bboxes_count': 0}






def main() -> None:
    """主函数（参考fix目录的通用实现模式）"""
    parser = argparse.ArgumentParser(description='应用 geometry_color 策略选择框和掩码，原子写回NPZ')
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='NPZ 根目录（如 /home/team/zouzhiyuan/dataset/sa1b/extracted）')
    parser.add_argument('--json-dir', type=str, required=True,
                        help='SA-1B JSON 目录（如 /home/team/zouzhiyuan/dataset/sa1b）')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='图片文件目录（如 /home/team/zouzhiyuan/dataset/sa1b）')
    parser.add_argument('--log', type=str, required=True, help='日志输出路径')
    parser.add_argument('--max-files', type=int, default=0, help='限制处理文件数，0 表示不限')
    parser.add_argument('--skip-if-flag', action='store_true',
                        help='若 NPZ 已存在 geometry_color_flag=1，则跳过')
    parser.add_argument('--max-instances', type=int, default=1,
                        help='最多选择的实例数（默认1）')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行工作进程数（默认：全部CPU核心）')
    
    args = parser.parse_args()
    
    npz_root = Path(args.npz_dir)
    json_root = Path(args.json_dir)
    image_root = Path(args.image_dir)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    npz_files = collect_npz_files(npz_root)
    if args.max_files and args.max_files > 0:
        npz_files = npz_files[:args.max_files]
    
    num_workers = args.workers if args.workers else cpu_count()
    print(f"📁 找到 {len(npz_files)} 个NPZ文件")
    print(f"🔧 使用 {num_workers} 个并行工作进程")
    print(f"⏱️  开始处理...")
    
    # 准备任务列表
    tasks = [
        (npz_path, json_root, image_root, args.max_instances, args.skip_if_flag)
        for npz_path in npz_files
    ]
    
    processed = 0
    skipped = 0
    missing = 0
    errors = 0
    
    t0 = time.time()
    
    # 使用进程池并行处理
    with open(log_path, 'w', encoding='utf-8') as log_f:
        with Pool(processes=num_workers) as pool:
            results = []
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_single_file, tasks):
                    results.append(result)
                    pbar.update(1)
                    
                    # 写入日志
                    if result['status'] == 'ok':
                        processed += 1
                        log_f.write(f'[OK] {result["stem"]} bboxes={result["bboxes_count"]}\n')
                    elif result['status'] == 'skip':
                        skipped += 1
                        if 'not found' in result['message']:
                            missing += 1
                        log_f.write(f'[SKIP] {result["stem"]}: {result["message"]}\n')
                    else:
                        errors += 1
                        log_f.write(f'[ERR] {result["stem"]}: {result["message"]}\n')
                    
                    log_f.flush()
    
    duration = time.time() - t0
    
    summary = (
        f'[DONE] processed={processed} skipped={skipped} missing={missing} '
        f'errors={errors} total={len(npz_files)} duration={duration:.1f}s '
        f'({duration/len(npz_files)*1000:.1f}ms/file)'
    )
    print(summary)




if __name__ == '__main__':
    main()


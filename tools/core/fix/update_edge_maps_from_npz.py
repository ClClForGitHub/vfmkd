#!/usr/bin/env python3
"""
从已有NPZ文件批量更新边缘图（使用Method B）
从NPZ文件名找到对应的JSON文件，用B方法生成边缘图，更新NPZ中的edge_64x64和edge_32x32
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils
import argparse
import json
import os
import time


def extract_edges_method_b(json_path, kernel_size=3):
    """
    使用Method B（优化版CPU）从JSON提取边缘图
    完全复刻extract_features_edge_comparison中的Method B实现
    
    Args:
        json_path: JSON标注文件路径
        kernel_size: 形态学操作核大小
        
    Returns:
        edge_maps: 字典，包含'original', 256, 64, 32的边缘图
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    height = data['image']['height']
    width = data['image']['width']
    annotations = data['annotations']
    
    # 创建形态学操作kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Method B：每个实例单独提取边缘后合并（与edge_comparison完全一致）
    combined_edge_map = np.zeros((height, width), dtype=np.uint8)
    
    if len(annotations) > 0:
        for ann in annotations:
            rle = ann['segmentation']
            mask = mask_utils.decode(rle)  # 从RLE解码
            
            # 对每个实例单独提取边缘
            edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            # 二值化并确保uint8类型（避免类型不匹配和溢出警告）
            edge = (edge > 0).astype(np.uint8)
            
            # 使用bitwise_or替代logical_or（直接在uint8上操作）
            combined_edge_map = np.bitwise_or(combined_edge_map, edge)
    
    # 生成多尺度边缘图（与extract_v1同款方法）
    edge_maps = {'original': combined_edge_map}
    for size in [256, 64, 32]:
        edge_float = combined_edge_map.astype(np.float32)
        edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
        edge_maps[size] = (edge_small > 0).astype(np.uint8)
    
    return edge_maps


def update_single_npz_edge_maps(npz_path, json_dir, kernel_size=3, set_edge_flag=True):
    """
    更新单个NPZ文件的边缘图
    
    Args:
        npz_path: NPZ文件路径
        json_dir: JSON文件目录
        kernel_size: 形态学操作核大小
        
    Returns:
        dict: 更新结果 {'success': bool, 'image_id': str, 'error': str or None}
    """
    try:
        # 从NPZ文件名提取image_id
        # sa_10000_features.npz -> sa_10000.json
        image_id = npz_path.stem.replace('_features', '')
        json_path = Path(json_dir) / f"{image_id}.json"
        
        if not json_path.exists():
            return {'success': False, 'image_id': image_id, 'error': f'JSON not found: {json_path}'}
        
        # 加载NPZ文件
        npz_data = dict(np.load(npz_path, allow_pickle=True))
        
        # 使用Method B生成新的边缘图
        edge_maps = extract_edges_method_b(json_path, kernel_size)
        
        # 更新NPZ中的边缘图（只更新64x64和32x32）
        npz_data['edge_64x64'] = edge_maps[64]
        npz_data['edge_32x32'] = edge_maps[32]
        
        # 可选：也更新256x256（如果存在）
        if 'edge_256x256' in npz_data:
            npz_data['edge_256x256'] = edge_maps[256]
        
        # 版本/标记写回，便于下次跳过
        if set_edge_flag:
            npz_data['edge_flag'] = np.array(1, dtype=np.uint8)
            npz_data['edge_version'] = np.array('B_v1')

        # 保存更新后的NPZ
        np.savez(npz_path, **npz_data)
        
        return {'success': True, 'image_id': image_id}
        
    except Exception as e:
        return {'success': False, 'image_id': npz_path.stem, 'error': str(e)}


def batch_update_edge_maps_from_npz(
    npz_dir,
    json_dir,
    kernel_size=3,
    max_files=None,
    sort_by='mtime',
    reverse=False,
    only_mtime_after=None,
    skip_if_processed=True,
    set_edge_flag=True,
):
    """
    批量更新NPZ文件中的边缘图
    
    Args:
        npz_dir: NPZ文件目录
        json_dir: JSON文件目录
        kernel_size: 形态学操作核大小
        max_files: 最大处理文件数（None表示全部）
        
    Returns:
        dict: 统计信息
    """
    npz_dir = Path(npz_dir)
    json_dir = Path(json_dir)
    
    # 获取所有NPZ文件（递归扫描子目录）
    npz_files = list(npz_dir.rglob("*_features.npz"))

    # 排序
    if sort_by == 'mtime':
        npz_files = sorted(npz_files, key=lambda p: p.stat().st_mtime, reverse=reverse)
    else:
        npz_files = sorted(npz_files, reverse=reverse)

    # 时间阈值过滤（仅处理修改时间更晚的文件）
    if only_mtime_after is not None:
        npz_files = [p for p in npz_files if p.stat().st_mtime > float(only_mtime_after)]

    # 截断最大数量
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"📁 找到 {len(npz_files)} 个NPZ文件")
    print(f"📁 JSON目录: {json_dir}")
    print(f"⏱️  开始批量更新边缘图（使用Method B）...")
    
    success_count = 0
    error_count = 0
    errors = []
    
    for npz_file in tqdm(npz_files, desc="Updating edge maps"):
        # 跳过已处理（仅当存在 edge_flag==1）
        if skip_if_processed:
            try:
                probe = np.load(npz_file, allow_pickle=True)
                edge_flag_ok = False
                if 'edge_flag' in probe:
                    try:
                        edge_flag_ok = int(probe['edge_flag']) == 1
                    except Exception:
                        edge_flag_ok = False
                probe.close()
                if edge_flag_ok:
                    continue
            except Exception:
                pass

        result = update_single_npz_edge_maps(npz_file, json_dir, kernel_size, set_edge_flag=set_edge_flag)
        
        if result['success']:
            success_count += 1
        else:
            error_count += 1
            errors.append(f"{result['image_id']}: {result['error']}")
    
    print(f"\n🎉 批量更新完成!")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {error_count} 个文件")
    
    if errors and len(errors) <= 10:
        print(f"\n❌ 错误详情:")
        for err in errors:
            print(f"  {err}")
    elif errors:
        print(f"\n❌ 前10个错误:")
        for err in errors[:10]:
            print(f"  {err}")
    
    return {
        'success_count': success_count,
        'error_count': error_count,
        'total': len(npz_files),
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description="批量更新NPZ文件中的边缘图（使用Method B）")
    parser.add_argument("--npz-dir", type=str, required=True, help="NPZ文件目录")
    parser.add_argument("--json-dir", type=str, required=True, help="JSON文件目录")
    parser.add_argument("--kernel-size", type=int, default=3, help="形态学操作核大小")
    parser.add_argument("--max-files", type=int, default=None, help="最大处理文件数（None=全部）")
    parser.add_argument("--sort-by", type=str, default="mtime", choices=["name", "mtime"], help="排序方式：name 或 mtime")
    parser.add_argument("--reverse", action="store_true", help="反向顺序（如mtime时即从新到旧）")
    parser.add_argument("--only-mtime-after", type=float, default=None, help="仅处理修改时间大于该epoch秒的文件")
    parser.add_argument("--skip-if-processed", action="store_true", default=True, help="已处理则跳过（仅检测edge_flag==1）")
    parser.add_argument("--no-skip-if-processed", dest="skip_if_processed", action="store_false", help="不跳过已处理")
    parser.add_argument("--set-edge-flag", action="store_true", default=True, help="写回edge_flag=1与edge_version=B_v1 标记")
    parser.add_argument("--no-set-edge-flag", dest="set_edge_flag", action="store_false", help="不写回标记")
    
    args = parser.parse_args()
    
    batch_update_edge_maps_from_npz(
        args.npz_dir,
        args.json_dir,
        args.kernel_size,
        args.max_files,
        sort_by=args.sort_by,
        reverse=args.reverse,
        only_mtime_after=args.only_mtime_after,
        skip_if_processed=args.skip_if_processed,
        set_edge_flag=args.set_edge_flag,
    )


if __name__ == "__main__":
    main()

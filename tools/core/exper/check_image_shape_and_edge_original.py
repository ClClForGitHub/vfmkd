#!/usr/bin/env python3
"""
详细检查 image_shape 和 edge_original 字段，判断是否可以处理成定长
"""

import tarfile
import numpy as np
import io
from pathlib import Path
from collections import defaultdict
import argparse


def check_fields_in_tar(tar_path: Path, max_samples: int = 100):
    """检查 tar 文件中的 image_shape 和 edge_original"""
    print(f"📦 检查 tar 文件: {tar_path}")
    
    image_shapes = []
    edge_original_shapes = []
    edge_original_present = 0
    edge_original_missing = 0
    
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            npz_members = [m for m in tar.getmembers() if m.name.endswith(".npz")]
            
            print(f"📊 找到 {len(npz_members)} 个 .npz 文件")
            print(f"🔍 检查前 {min(max_samples, len(npz_members))} 个样本...\n")
            
            for idx, member in enumerate(npz_members[:max_samples]):
                if idx % 20 == 0 and idx > 0:
                    print(f"  处理进度: {idx}/{min(max_samples, len(npz_members))}")
                
                npz_file_obj = tar.extractfile(member)
                if npz_file_obj is None:
                    continue
                
                npz_bytes = npz_file_obj.read()
                
                try:
                    with np.load(io.BytesIO(npz_bytes), allow_pickle=True) as data:
                        # 检查 image_shape
                        if 'image_shape' in data.files:
                            img_shape = data['image_shape']
                            image_shapes.append({
                                'shape': img_shape.shape,
                                'value': tuple(img_shape) if img_shape.ndim > 0 else img_shape.item(),
                                'dtype': str(img_shape.dtype),
                                'size': img_shape.size,
                            })
                        
                        # 检查 edge_original
                        if 'edge_original' in data.files:
                            edge_orig = data['edge_original']
                            edge_original_shapes.append({
                                'shape': edge_orig.shape,
                                'dtype': str(edge_orig.dtype),
                                'size': edge_orig.size,
                                'sample_id': member.name,
                            })
                            edge_original_present += 1
                        else:
                            edge_original_missing += 1
                            
                except Exception as e:
                    print(f"  ⚠️  处理 {member.name} 时出错: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        return
    
    # 分析结果
    print("\n" + "="*80)
    print("📊 image_shape 分析结果")
    print("="*80)
    
    if image_shapes:
        unique_shapes = set(str(s['shape']) for s in image_shapes)
        unique_values = set(str(s['value']) for s in image_shapes)
        unique_dtypes = set(s['dtype'] for s in image_shapes)
        
        print(f"\n✅ 统计:")
        print(f"  - 总样本数: {len(image_shapes)}")
        print(f"  - 唯一形状数: {len(unique_shapes)}")
        print(f"  - 唯一值数: {len(unique_values)}")
        print(f"  - 唯一类型数: {len(unique_dtypes)}")
        
        print(f"\n📐 形状信息:")
        for shape_str in unique_shapes:
            print(f"  - {shape_str}")
        
        print(f"\n💡 结论:")
        if len(unique_shapes) == 1:
            shape_str = list(unique_shapes)[0]
            if shape_str == "(3,)" or shape_str == "()":
                print(f"  ✅ image_shape 形状固定: {shape_str}")
                print(f"  ✅ 可以按定长处理：每个样本固定 {shape_str} 大小")
                print(f"  ✅ 存储方式：直接拼接，每个样本占用固定字节数")
                
                # 计算固定大小
                sample = image_shapes[0]
                if sample['shape'] == (3,):
                    # 3个整数
                    if 'int32' in sample['dtype']:
                        fixed_size = 3 * 4  # 12 bytes
                    elif 'int64' in sample['dtype']:
                        fixed_size = 3 * 8  # 24 bytes
                    else:
                        fixed_size = 3 * 4  # 默认 int32
                    print(f"  ✅ 固定大小: {fixed_size} bytes/样本")
                elif sample['shape'] == ():
                    # 标量，但存储为数组
                    fixed_size = 3 * 4  # 12 bytes
                    print(f"  ✅ 固定大小: {fixed_size} bytes/样本（转换为 (3,) 数组）")
            else:
                print(f"  ⚠️  形状不固定: {shape_str}")
        else:
            print(f"  ⚠️  形状不固定，有 {len(unique_shapes)} 种形状")
        
        print(f"\n📋 示例值（前10个）:")
        for i, s in enumerate(image_shapes[:10]):
            print(f"  [{i}] shape={s['shape']}, value={s['value']}, dtype={s['dtype']}")
    else:
        print("  ❌ 未找到 image_shape 字段")
    
    print("\n" + "="*80)
    print("📊 edge_original 分析结果")
    print("="*80)
    
    print(f"\n✅ 统计:")
    print(f"  - 总样本数: {len(image_shapes) if image_shapes else max_samples}")
    print(f"  - 包含 edge_original: {edge_original_present}")
    print(f"  - 不包含 edge_original: {edge_original_missing}")
    print(f"  - 存在率: {edge_original_present / (edge_original_present + edge_original_missing) * 100:.1f}%")
    
    if edge_original_shapes:
        unique_shapes = set(str(s['shape']) for s in edge_original_shapes)
        unique_dtypes = set(s['dtype'] for s in edge_original_shapes)
        
        print(f"\n📐 形状信息:")
        print(f"  - 唯一形状数: {len(unique_shapes)}")
        print(f"  - 唯一类型数: {len(unique_dtypes)}")
        
        print(f"\n📋 所有形状:")
        shape_counts = defaultdict(int)
        for s in edge_original_shapes:
            shape_counts[str(s['shape'])] += 1
        
        for shape_str, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
            print(f"  - {shape_str}: {count} 次")
        
        print(f"\n💡 结论:")
        if len(unique_shapes) == 1:
            shape_str = list(unique_shapes)[0]
            print(f"  ✅ edge_original 形状固定: {shape_str}")
            print(f"  ✅ 可以按定长处理：每个样本固定 {shape_str} 大小")
            
            sample = edge_original_shapes[0]
            if 'uint8' in sample['dtype']:
                h, w = sample['shape']
                fixed_size = h * w * 1  # uint8 = 1 byte
                print(f"  ✅ 固定大小: {fixed_size} bytes/样本 ({h}×{w} uint8)")
            else:
                print(f"  ⚠️  类型: {sample['dtype']}")
        else:
            print(f"  ⚠️  edge_original 形状不固定，有 {len(unique_shapes)} 种形状")
            print(f"  💡 处理方案:")
            print(f"     1. 如果不需要原始尺寸，可以完全忽略此字段")
            print(f"     2. 如果需要，可以统一 resize 到固定尺寸（如 1024×1024）")
            print(f"     3. 或者保持变长，使用索引文件")
            
            # 检查是否可以统一到某个尺寸
            all_shapes = [tuple(s['shape']) for s in edge_original_shapes]
            if all(len(s) == 2 for s in all_shapes):
                heights = [s[0] for s in all_shapes]
                widths = [s[1] for s in all_shapes]
                max_h, max_w = max(heights), max(widths)
                min_h, min_w = min(heights), min(widths)
                print(f"\n     尺寸范围:")
                print(f"       - 高度: {min_h} ~ {max_h}")
                print(f"       - 宽度: {min_w} ~ {max_w}")
                print(f"       - 如果统一到 {max_h}×{max_w}，需要 {max_h * max_w} bytes/样本")
    else:
        print("  ⚠️  未找到 edge_original 字段（或所有样本都没有）")
        print("  💡 如果此字段不是必需的，可以完全忽略，不存储")
    
    print("\n" + "="*80)
    print("🎯 最终建议")
    print("="*80)
    
    # image_shape 建议
    if image_shapes and len(set(str(s['shape']) for s in image_shapes)) == 1:
        print("\n✅ image_shape:")
        print("  - 可以按定长处理")
        print("  - 存储方式: image_shapes.bin（直接拼接，每个样本 12 bytes）")
        print("  - 读取方式: offset = sample_id * 12，读取 12 bytes，解析为 (3,) int32")
    
    # edge_original 建议
    if edge_original_shapes:
        if len(set(str(s['shape']) for s in edge_original_shapes)) == 1:
            print("\n✅ edge_original:")
            print("  - 可以按定长处理")
            sample = edge_original_shapes[0]
            h, w = sample['shape']
            size = h * w
            print(f"  - 存储方式: edge_original.bin（直接拼接，每个样本 {size} bytes）")
            print(f"  - 读取方式: offset = sample_id * {size}，读取 {size} bytes，reshape 为 ({h}, {w})")
        else:
            print("\n⚠️  edge_original:")
            print("  - 形状不固定，建议:")
            print("    方案1: 如果不需要，完全忽略（推荐）")
            print("    方案2: 统一 resize 到固定尺寸（如 1024×1024）")
            print("    方案3: 保持变长，使用索引文件")
    else:
        print("\n✅ edge_original:")
        print("  - 大多数样本都没有此字段")
        print("  - 建议: 完全忽略，不存储（推荐）")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="检查 image_shape 和 edge_original 字段，判断是否可以处理成定长"
    )
    parser.add_argument(
        'tar_path',
        type=Path,
        nargs='?',
        help='tar 文件路径（可选，默认使用第一个 shard）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='最大检查样本数（默认: 100）'
    )
    parser.add_argument(
        '--shard-dir',
        type=Path,
        help='shard 目录（如果未指定 tar_path，则从该目录查找第一个 .tar 文件）'
    )
    
    args = parser.parse_args()
    
    # 确定 tar 文件路径
    tar_path = args.tar_path
    if tar_path is None:
        if args.shard_dir:
            shard_dir = args.shard_dir
        else:
            # 默认路径
            shard_dir = Path("/home/team/zouzhiyuan/dataset/sa1b_tar_shards")
        
        if shard_dir.exists():
            tar_files = sorted(list(shard_dir.glob("*.tar*")))
            if tar_files:
                tar_path = tar_files[0]
                print(f"📁 未指定 tar 文件，使用第一个: {tar_path}")
            else:
                print(f"❌ 错误：在 {shard_dir} 中未找到 .tar 文件")
                return
        else:
            print(f"❌ 错误：目录不存在: {shard_dir}")
            return
    
    check_fields_in_tar(tar_path, max_samples=args.max_samples)


if __name__ == "__main__":
    main()


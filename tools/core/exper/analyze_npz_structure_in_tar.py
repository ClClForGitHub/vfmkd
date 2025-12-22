#!/usr/bin/env python3
"""
分析 tar 文件中的 npz 文件结构，判断数据是定长还是变长的。

用途：
- 检查每个 npz 文件中的键（keys）
- 分析每个键的数据形状（shape）和类型（dtype）
- 判断数据是定长还是变长
- 为后续的二进制存储格式设计提供依据
"""

import tarfile
import numpy as np
import io
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import argparse


def analyze_npz_structure(npz_bytes: bytes, npz_name: str) -> Dict[str, Any]:
    """分析单个 npz 文件的结构"""
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=True) as data:
            result = {
                'name': npz_name,
                'keys': list(data.files),
                'key_info': {},
                'has_variable_length': False,
            }
            
            for key in data.files:
                try:
                    arr = data[key]
                    key_info = {
                        'dtype': str(arr.dtype),
                        'shape': arr.shape,
                        'size': arr.size,
                        'is_object': arr.dtype == object,
                        'is_variable_length': False,
                    }
                    
                    # 检查是否是变长数据
                    if arr.dtype == object:
                        # object 类型通常是变长的（如列表、不同形状的数组）
                        key_info['is_variable_length'] = True
                        result['has_variable_length'] = True
                        
                        # 尝试分析 object 数组的内容
                        if arr.size > 0:
                            first_item = arr.item(0)
                            if isinstance(first_item, np.ndarray):
                                key_info['first_item_shape'] = first_item.shape
                                key_info['first_item_dtype'] = str(first_item.dtype)
                            else:
                                key_info['first_item_type'] = str(type(first_item))
                    else:
                        # 检查是否是标量或固定形状数组
                        if arr.ndim == 0:
                            # 标量
                            key_info['value'] = arr.item()
                        elif arr.ndim > 0:
                            # 数组，检查是否有明显的变长特征
                            # 例如：一维数组的长度可能不同
                            if arr.ndim == 1:
                                # 一维数组可能是变长的（如不同数量的框）
                                key_info['is_variable_length'] = True
                                result['has_variable_length'] = True
                    
                    result['key_info'][key] = key_info
                    
                except Exception as e:
                    result['key_info'][key] = {
                        'error': str(e)
                    }
            
            return result
            
    except Exception as e:
        return {
            'name': npz_name,
            'error': str(e)
        }


def analyze_tar_file(tar_path: Path, max_samples: int = 100) -> Dict[str, Any]:
    """分析 tar 文件中的所有 npz 文件"""
    print(f"📦 分析 tar 文件: {tar_path}")
    
    if not tar_path.exists():
        print(f"❌ 错误：文件不存在: {tar_path}")
        return {}
    
    results = []
    key_statistics = defaultdict(list)  # 统计每个键的形状信息
    
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            npz_members = [m for m in tar.getmembers() if m.name.endswith(".npz")]
            
            if not npz_members:
                print("❌ 未找到 .npz 文件")
                return {}
            
            print(f"📊 找到 {len(npz_members)} 个 .npz 文件")
            
            # 限制分析的样本数量
            sample_count = min(max_samples, len(npz_members))
            print(f"🔍 分析前 {sample_count} 个样本...")
            
            for idx, member in enumerate(npz_members[:sample_count]):
                if idx % 10 == 0:
                    print(f"  处理进度: {idx+1}/{sample_count}")
                
                npz_file_obj = tar.extractfile(member)
                if npz_file_obj is None:
                    continue
                
                npz_bytes = npz_file_obj.read()
                result = analyze_npz_structure(npz_bytes, member.name)
                results.append(result)
                
                # 收集统计信息
                if 'key_info' in result:
                    for key, info in result['key_info'].items():
                        if 'shape' in info:
                            key_statistics[key].append({
                                'shape': info['shape'],
                                'dtype': info['dtype'],
                                'is_variable_length': info.get('is_variable_length', False),
                            })
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        return {}
    
    # 分析统计结果
    analysis = {
        'total_samples': len(results),
        'key_statistics': {},
        'summary': {},
    }
    
    for key, stats in key_statistics.items():
        shapes = [s['shape'] for s in stats]
        dtypes = [s['dtype'] for s in stats]
        is_variable = [s['is_variable_length'] for s in stats]
        
        # 检查形状是否一致
        unique_shapes = set(str(s) for s in shapes)
        unique_dtypes = set(dtypes)
        has_variable_length = any(is_variable)
        
        analysis['key_statistics'][key] = {
            'count': len(stats),
            'unique_shapes': list(unique_shapes),
            'unique_dtypes': list(unique_dtypes),
            'is_variable_length': has_variable_length,
            'is_fixed_length': len(unique_shapes) == 1 and not has_variable_length,
            'sample_shapes': shapes[:5],  # 前5个样本的形状
        }
    
    # 生成摘要
    fixed_keys = []
    variable_keys = []
    for key, stats in analysis['key_statistics'].items():
        if stats['is_fixed_length']:
            fixed_keys.append(key)
        else:
            variable_keys.append(key)
    
    analysis['summary'] = {
        'fixed_length_keys': fixed_keys,
        'variable_length_keys': variable_keys,
        'total_keys': len(analysis['key_statistics']),
    }
    
    return {
        'tar_path': str(tar_path),
        'results': results,
        'analysis': analysis,
    }


def print_analysis_report(analysis_result: Dict[str, Any]):
    """打印分析报告"""
    if not analysis_result:
        print("❌ 没有分析结果")
        return
    
    print("\n" + "="*80)
    print("📊 NPZ 数据结构分析报告")
    print("="*80)
    
    analysis = analysis_result['analysis']
    summary = analysis['summary']
    
    print(f"\n📈 摘要:")
    print(f"  - 分析样本数: {analysis['total_samples']}")
    print(f"  - 总键数: {summary['total_keys']}")
    print(f"  - 定长键数: {len(summary['fixed_length_keys'])}")
    print(f"  - 变长键数: {len(summary['variable_length_keys'])}")
    
    print(f"\n✅ 定长键 (Fixed Length):")
    if summary['fixed_length_keys']:
        for key in summary['fixed_length_keys']:
            stats = analysis['key_statistics'][key]
            print(f"  - {key:<30} 形状: {stats['unique_shapes'][0]}, 类型: {stats['unique_dtypes'][0]}")
    else:
        print("  (无)")
    
    print(f"\n⚠️  变长键 (Variable Length):")
    if summary['variable_length_keys']:
        for key in summary['variable_length_keys']:
            stats = analysis['key_statistics'][key]
            print(f"  - {key:<30} 形状变化: {len(stats['unique_shapes'])} 种")
            print(f"    示例形状: {stats['sample_shapes'][:3]}")
            if stats['is_variable_length']:
                print(f"    原因: 数据本身是变长的（object 类型或一维数组）")
    else:
        print("  (无)")
    
    print(f"\n📋 详细统计:")
    for key, stats in analysis['key_statistics'].items():
        print(f"\n  {key}:")
        print(f"    - 出现次数: {stats['count']}")
        print(f"    - 唯一形状数: {len(stats['unique_shapes'])}")
        print(f"    - 唯一类型数: {len(stats['unique_dtypes'])}")
        print(f"    - 是否定长: {stats['is_fixed_length']}")
        if len(stats['unique_shapes']) <= 5:
            print(f"    - 所有形状: {stats['unique_shapes']}")
        else:
            print(f"    - 前5个形状: {stats['sample_shapes']}")
    
    print("\n" + "="*80)
    print("💡 存储格式建议:")
    print("="*80)
    
    if summary['fixed_length_keys']:
        print("\n✅ 定长数据可以直接拼接存储（类似 EdgeSAM 方案）:")
        for key in summary['fixed_length_keys']:
            stats = analysis['key_statistics'][key]
            shape_str = stats['unique_shapes'][0]
            print(f"  - {key}: {shape_str}")
    
    if summary['variable_length_keys']:
        print("\n⚠️  变长数据需要索引文件（记录 offset 和 length）:")
        for key in summary['variable_length_keys']:
            print(f"  - {key}")
        print("\n  建议方案:")
        print("  1. 将不同键拆分到不同的 .bin 文件（列式存储）")
        print("  2. 定长键: features.bin, edge_*.bin 等")
        print("  3. 变长键: bboxes.bin, masks.bin 等 + 索引文件")
        print("  4. 索引文件格式: sample_id, offset, length (每个变长键一个索引)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="分析 tar 文件中的 npz 文件结构，判断数据是定长还是变长"
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
        help='最大分析样本数（默认: 100）'
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
                sys.exit(1)
        else:
            print(f"❌ 错误：目录不存在: {shard_dir}")
            sys.exit(1)
    
    # 执行分析
    analysis_result = analyze_tar_file(tar_path, max_samples=args.max_samples)
    
    # 打印报告
    print_analysis_report(analysis_result)
    
    # 保存结果到文件
    output_path = Path("npz_structure_analysis.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("NPZ 数据结构分析报告\n")
        f.write("="*80 + "\n\n")
        
        if analysis_result:
            analysis = analysis_result['analysis']
            summary = analysis['summary']
            
            f.write(f"分析样本数: {analysis['total_samples']}\n")
            f.write(f"总键数: {summary['total_keys']}\n")
            f.write(f"定长键数: {len(summary['fixed_length_keys'])}\n")
            f.write(f"变长键数: {len(summary['variable_length_keys'])}\n\n")
            
            f.write("定长键:\n")
            for key in summary['fixed_length_keys']:
                stats = analysis['key_statistics'][key]
                f.write(f"  {key}: {stats['unique_shapes'][0]}\n")
            
            f.write("\n变长键:\n")
            for key in summary['variable_length_keys']:
                stats = analysis['key_statistics'][key]
                f.write(f"  {key}: {len(stats['unique_shapes'])} 种形状\n")
    
    print(f"\n💾 详细报告已保存到: {output_path}")


if __name__ == "__main__":
    main()


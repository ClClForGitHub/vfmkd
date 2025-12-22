#!/usr/bin/env python3
"""
SA-1B TAR Sharding Tool (Extreme Performance Edition)

针对 100核+ CPU 和高速存储优化的打包工具。

主要优化：
1. 使用 os.scandir 替代 glob，速度提升明显
2. 全程使用 str 替代 Path 对象，大幅降低内存占用和多进程序列化开销
3. tarfile 写入开启 4MB 缓冲区，优化 I/O吞吐
4. 禁用不必要的 GC，极致压榨 CPU

用法:
    python create_sa1b_tar_shards.py \
        --features-dir /path/to/npz \
        --images-dir /path/to/jpg \
        --output-dir /path/to/shards \
        --workers 80
"""

import argparse
import os
import re
import tarfile
import time
import gc
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Set, Tuple
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ==========================================
# 核心工作函数 (Worker)
# ==========================================

def _write_one_shard(args):
    """
    多进程 Worker: 写入单个 Tar Shard
    
    Args:
        args: (shard_idx, shard_data, output_dir, mode, suffix)
            shard_data: [(npz_path_str, img_path_str, image_id_str), ...]
    """
    shard_idx, shard_data, output_dir, mode, suffix, overwrite = args
    
    # 构造输出文件名
    shard_name = f"sa1b_shard_{shard_idx:05d}{suffix}"
    shard_path = os.path.join(output_dir, shard_name)
    
    count = 0
    
    if (not overwrite) and os.path.exists(shard_path):
        return shard_idx, 0, "skip_existing"

    try:
        # 开启大缓冲区 (4MB)，这对大批量小文件写入至关重要
        # format=tarfile.PAX_FORMAT 支持长文件名和大于8GB的文件
        with tarfile.open(shard_path, mode, bufsize=4*1024*1024, format=tarfile.PAX_FORMAT) as tar:
            for npz_path, img_path, image_id in shard_data:
                # 1. 添加 NPZ
                # 保持原名: sa_12345_features.npz
                npz_arcname = os.path.basename(npz_path)
                tar.add(npz_path, arcname=npz_arcname)
                
                # 2. 添加 JPG
                # 规范化命名: sa_12345.jpg
                # image_id 已经是规范化的 key (sa_xxxx)
                img_arcname = f"{image_id}.jpg"
                tar.add(img_path, arcname=img_arcname)
                
                count += 1
                
        return shard_idx, count, None
    except Exception as e:
        return shard_idx, 0, str(e)


# ==========================================
# 高性能扫描与索引函数
# ==========================================

def fast_scan_images(images_dir: str) -> Dict[str, str]:
    """
    使用 os.scandir 快速扫描目录并建立索引。
    
    Returns:
        { 'sa_12345': '/full/path/to/sa_12345.jpg' }
    """
    print(f"[Scanning] 正在扫描图像目录: {images_dir} ...")
    index = {}
    
    try:
        # 临时禁用 GC 加速大量对象的创建
        gc.disable()
        
        # os.scandir 是目前 Python 最快的文件遍历方式
        with os.scandir(images_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.jpg'):
                    # 解析文件名 sa_123.jpg -> sa_123
                    # 避免使用 splitext，直接切片更快
                    name = entry.name
                    stem = name[:-4]  # 去掉 .jpg
                    
                    # 规范化 Key: 确保是 sa_ 开头
                    if stem.startswith('sa_'):
                        key = stem
                    else:
                        key = 'sa_' + stem
                        
                    index[key] = entry.path
    finally:
        gc.enable()
        
    print(f"[Index] 图像扫描完成，索引大小: {len(index)}")
    return index


def fast_scan_npz(features_dir: str) -> List[str]:
    """
    快速扫描 NPZ 文件列表
    
    Returns:
        ['/full/path/to/sa_123_features.npz', ...]
    """
    print(f"[Scanning] 正在扫描特征目录: {features_dir} ...")
    npz_files = []
    
    try:
        gc.disable()
        with os.scandir(features_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('_features.npz'):
                    npz_files.append(entry.path)
    finally:
        gc.enable()
    
    # 排序以保证 Shard 内容的确定性
    npz_files.sort()
    print(f"[Index] 特征扫描完成，共发现: {len(npz_files)} 个 NPZ")
    return npz_files


def match_pairs(npz_files: List[str], image_index: Dict[str, str], max_samples: int = None) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    匹配 NPZ 和 JPG
    
    Returns:
        (pairs, missing)
        pairs: [(npz_path_str, img_path_str, image_id_str), ...]
        missing: [image_id_str, ...]
    """
    pairs = []
    missing = []
    
    print("[Matching] 开始匹配 NPZ 和 JPG...")
    
    iterator = npz_files
    if max_samples:
        iterator = npz_files[:max_samples]
    
    for npz_path in tqdm(iterator, desc="Matching", unit="file"):
        # 从路径提取文件名: /path/to/sa_123_features.npz -> sa_123
        filename = os.path.basename(npz_path)
        # 去掉 _features.npz (长度为 13)
        image_id = filename[:-13]
        
        img_path = image_index.get(image_id)
        
        if img_path:
            pairs.append((npz_path, img_path, image_id))
        else:
            missing.append(image_id)
            
    print(f"[Result] 匹配成功: {len(pairs)}, 缺失图像: {len(missing)}")
    if missing and len(missing) <= 10:
        print(f"[WARN] 缺失 image_id 示例: {missing[:10]}")
    
    return pairs, missing


# ==========================================
# 主逻辑
# ==========================================

def _scan_existing_shards(output_dir: str) -> Set[int]:
    """
    扫描已存在的 shard，返回其索引集合（如 0, 1, 2 ...）
    """
    if not os.path.isdir(output_dir):
        return set()

    shard_indices: Set[int] = set()
    pattern = re.compile(r"sa1b_shard_(\d+)\.tar(\.\w+)?$")

    with os.scandir(output_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match:
                shard_indices.add(int(match.group(1)))

    return shard_indices


def create_tar_shards(
    features_dir: str,
    images_dir: str,
    output_dir: str,
    shard_size: int = 1024,
    compress: str | None = None,
    max_samples: int | None = None,
    workers: int = 8,
    overwrite_existing: bool = False,
) -> None:
    """
    创建 tar shards
    
    Args:
        features_dir: NPZ 特征文件目录（字符串路径）
        images_dir: Resize 后的 JPG 图像目录（字符串路径）
        output_dir: 输出 Tar Shard 目录（字符串路径）
        shard_size: 每个 Shard 的样本数
        compress: 压缩格式 ("gz", "bz2", "xz" 或 None)
        max_samples: 调试用: 最大处理样本数
        workers: 工作进程数
    """
    t0 = time.time()
    
    # 1. 建立索引 (单线程极速扫描)
    # 对于单个大目录，Python的多线程/多进程扫描由于GIL和OS锁，往往不如单线程 scandir 快
    image_index = fast_scan_images(images_dir)
    npz_files = fast_scan_npz(features_dir)
    
    # 2. 匹配
    pairs, missing = match_pairs(npz_files, image_index, max_samples)
    if not pairs:
        raise RuntimeError("没有找到任何 npz + jpg 配对，请检查路径和命名规则。")
    
    # 3. 准备分片任务
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算 Shard 数量
    num_shards = (len(pairs) + shard_size - 1) // shard_size
    print(f"[Plan] 计划生成 {num_shards} 个 Shards, 每个包含约 {shard_size} 个样本")

    existing_indices = _scan_existing_shards(output_dir)
    if existing_indices and not overwrite_existing:
        print(f"[Resume] 检测到 {len(existing_indices)} 个已存在的 shard，将自动跳过同名文件（使用 --overwrite-existing 可覆盖）")
    
    # 确定压缩模式 (默认不压缩 'w' 以追求最大 I/O 吞吐)
    mode = "w"
    suffix = ".tar"
    if compress:
        c = compress.lower()
        if c == "gz":
            mode, suffix = "w:gz", ".tar.gz"
        elif c == "bz2":
            mode, suffix = "w:bz2", ".tar.bz2"
        elif c in ("xz", "lzma"):
            mode, suffix = "w:xz", ".tar.xz"
        else:
            raise ValueError(f"不支持的压缩格式: {compress}")
    
    # 生成任务列表
    tasks = []
    total_target_samples = 0
    skipped_samples = 0

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size
        shard_data = pairs[start_idx:end_idx]
        if not shard_data:
            continue

        if (not overwrite_existing) and (i in existing_indices):
            skipped_samples += len(shard_data)
            continue

        tasks.append((i, shard_data, output_dir, mode, suffix, overwrite_existing))
        total_target_samples += len(shard_data)

    if not tasks:
        print("[Plan] 所有 shard 已存在，未发现需要写入的任务。")
        return

    if skipped_samples:
        print(f"[Resume] 自动跳过 {skipped_samples} 条已存在的样本（对应 {len(existing_indices & set(range(num_shards)))} 个 shard）")
    
    # 4. 多进程执行
    # 使用 min(workers, num_shards) 避免创建无用的进程
    real_workers = max(1, min(workers, len(tasks)))
    print(f"[Exec] 启动 {real_workers} 个 Worker 进程进行并行打包...")
    
    total_written = 0
    errors = []
    
    # 使用 imap_unordered 实现更平滑的进度条更新
    with mp.Pool(processes=real_workers) as pool:
        pbar = tqdm(total=total_target_samples, desc="Writing Shards", unit="sample")
        
        for idx, count, error in pool.imap_unordered(_write_one_shard, tasks):
            if error and error != "skip_existing":
                errors.append((idx, error))
                print(f"\n❌ Shard {idx:05d} 失败: {error}")
            else:
                total_written += count
                pbar.update(count)
        
        pbar.close()
    
    t1 = time.time()
    duration = t1 - t0
    speed = total_written / duration if duration > 0 else 0
    
    print("\n" + "="*50)
    print(f"✅ 打包完成!")
    print(f"📊 总样本数: {total_written}")
    print(f"⏱️  总耗时: {duration:.2f} 秒")
    print(f"🚀 平均速度: {speed:.2f} samples/s")
    print(f"📁 输出目录: {output_dir}")
    if errors:
        print(f"⚠️  失败 shard 数: {len(errors)}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="SA-1B TAR Sharding (Extreme Optimized)")
    parser.add_argument("--features-dir", type=str, required=True,
                        help="NPZ 特征文件目录，例如 /home/.../sa1b/extracted")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Resize 后的 JPG 图像目录，例如 /home/.../sa1b_resized_1024")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出 Tar Shard 目录，例如 /home/.../sa1b_tar_shards")
    
    # 性能相关默认值调整
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="每个 Shard 的样本数（默认 1000，建议 1000~2000）")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help=f"工作进程数 (默认: {mp.cpu_count()} 核心，建议 80% CPU 核心数)")
    parser.add_argument("--compress", type=str, default=None,
                        choices=["gz", "bz2", "xz"],
                        help="压缩格式 (建议留空以获得最高速度)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="调试用: 最大处理样本数")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="若目标文件已存在，强制覆盖（默认跳过已存在的 shard）")
    
    args = parser.parse_args()
    
    # 路径检查
    if not os.path.isdir(args.features_dir):
        print(f"❌ 错误: 特征目录不存在 {args.features_dir}")
        return
    if not os.path.isdir(args.images_dir):
        print(f"❌ 错误: 图像目录不存在 {args.images_dir}")
        return
    
    create_tar_shards(
        features_dir=args.features_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        compress=args.compress,
        max_samples=args.max_samples,
        workers=args.workers,
        overwrite_existing=args.overwrite_existing,
    )


if __name__ == "__main__":
    # 设置启动方法为 fork (Linux默认) 或 spawn (更安全但慢)，这里不做强制限制
    # 如果遇到 deadlock 问题，可以尝试取消注释下面这行:
    # mp.set_start_method('spawn')
    main()

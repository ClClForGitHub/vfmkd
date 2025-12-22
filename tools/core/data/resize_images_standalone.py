#!/usr/bin/env python3
"""
SA-1B 高质量独立 Resize 脚本 (High Quality Resize Standalone)

环境: RTX 3090 + 100核 CPU

功能:
将指定目录下的所有图片 Resize 到 1024x1024，并以最高质量 JPG 保存。

逻辑与 extract_features_v5.py 完全一致。

特性:
1. [对齐训练] 使用 PIL.Image.resize(..., Image.BILINEAR)
2. [最高画质] 保存参数 quality=100, subsampling=0 (4:4:4)
3. [极致性能] 多进程并行处理，自动利用所有 CPU 核心
4. [断点续传] 自动跳过已存在的文件
5. [目录保持] 保持原始目录结构
"""

import os
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# =========================================================================
# 核心处理函数 (Worker)
# =========================================================================

def _resize_worker(args):
    """
    单张图片处理函数
    
    Args:
        args: (src_path, dst_path, target_size)
    
    Returns:
        int: 1=成功, 0=跳过(已存在), -1=失败
    """
    src_path, dst_path, target_size = args
    
    try:
        # 断点续传：如果目标文件已存在，直接跳过
        if os.path.exists(dst_path):
            return 0
        
        # 1. 使用 PIL 加载 (对齐训练读取方式)
        with Image.open(src_path) as img:
            # 强制转为 RGB，防止 RGBA 或 Grayscale 导致保存 JPEG 失败或通道不一致
            img = img.convert('RGB')
            
            # 2. Resize (对齐训练插值方式)
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.BILINEAR)
            
            # 3. 创建父目录 (多进程即使竞争也安全)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 4. 保存 (最高画质配置)
            # quality=100: 关闭压缩伪影
            # subsampling=0: 4:4:4 采样，保留所有色度信息，边缘更锐利
            img.save(dst_path, format='JPEG', quality=100, subsampling=0)
        
        return 1  # Success
    except Exception as e:
        # 可以在这里打印错误日志，但在大规模处理时建议静默或记录到文件
        # print(f"Error processing {src_path}: {e}")
        return -1  # Error


# =========================================================================
# 主函数
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SA-1B High-Quality Resize Tool (与 extract_features_v5.py 逻辑完全一致)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. 基本使用:
     python resize_images_standalone.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/resized

  2. 自定义尺寸和进程数:
     python resize_images_standalone.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/resized \\
        --target-size 1024 \\
        --num-workers 32

  3. 使用所有CPU核心:
     python resize_images_standalone.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/resized \\
        --num-workers $(nproc)
        """
    )
    
    parser.add_argument("--input-dir", type=str, required=True,
                       help="原始图片根目录（支持递归扫描子目录）")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出图片根目录（保持原始目录结构）")
    parser.add_argument("--target-size", type=int, default=1024,
                       help="目标分辨率 (默认 1024)")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                       help=f"进程数 (默认使用所有核心: {mp.cpu_count()})")
    
    args = parser.parse_args()
    
    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    
    if not src_dir.exists():
        print(f"❌ 错误: 输入目录不存在 {src_dir}")
        return
    
    print(f"\n{'='*60}")
    print("🚀 启动高性能 Resize (PIL Backend)")
    print(f"{'='*60}")
    print(f"📂 输入: {src_dir}")
    print(f"📂 输出: {dst_dir}")
    print(f"⚙️  设置: Size={args.target_size}x{args.target_size}")
    print(f"🎨 质量: Quality=100, Subsampling=0 (4:4:4 原色采样)")
    print(f"🔥 进程: {args.num_workers}")
    print(f"{'='*60}\n")
    
    # 1. 扫描文件
    print("🔍 正在扫描文件列表...")
    # 支持常见图片格式（大小写不敏感）
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tif', '*.tiff',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP', '*.TIF', '*.TIFF']
    files = []
    for ext in exts:
        # rglob 递归查找所有子目录
        files.extend(list(src_dir.rglob(ext)))
    
    total_files = len(files)
    print(f"📋 找到 {total_files} 张图片")
    
    if total_files == 0:
        print("❌ 未找到图片，请检查路径")
        return
    
    # 2. 构建任务列表
    print("📝 构建任务列表...")
    tasks = []
    for p in files:
        # 保持相对路径结构
        rel_path = p.relative_to(src_dir)
        # 强制修改后缀为 .jpg
        out_p = (dst_dir / rel_path).with_suffix('.jpg')
        tasks.append((str(p), str(out_p), args.target_size))
    
    print(f"✅ 任务列表构建完成，共 {len(tasks)} 个任务\n")
    
    # 3. 并行处理
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 使用 imap_unordered 实现流式处理，进度条更平滑，内存占用更低
    with mp.Pool(args.num_workers) as pool:
        pbar = tqdm(total=total_files, unit="img", desc="Processing", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # chunksize=10 可以减少进程间通信开销，对于小任务有帮助
        for res in pool.imap_unordered(_resize_worker, tasks, chunksize=10):
            if res == 1:
                success_count += 1
            elif res == 0:
                skip_count += 1
            else:
                error_count += 1
            pbar.update(1)
        
        pbar.close()
    
    # 4. 输出统计信息
    print(f"\n{'='*60}")
    print("✅ 处理完成!")
    print(f"{'='*60}")
    print(f"🟢 成功: {success_count} 张")
    print(f"🟡 跳过: {skip_count} 张 (已存在)")
    print(f"🔴 失败: {error_count} 张")
    
    if success_count > 0:
        print(f"\n📊 统计:")
        print(f"   成功率: {success_count/total_files*100:.1f}%")
        print(f"   跳过率: {skip_count/total_files*100:.1f}%")
        if error_count > 0:
            print(f"   失败率: {error_count/total_files*100:.1f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 确保在 Linux/Windows 下多进程行为一致且安全
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    main()


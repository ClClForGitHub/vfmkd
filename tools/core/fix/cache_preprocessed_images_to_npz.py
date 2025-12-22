#!/usr/bin/env python3
"""
高效的并行图像预处理脚本。

读取 --images-dir 中的所有图像，将其 resize 到 --input-size，
并保存到 --output-dir。

它使用 multiprocessing 来利用所有 CPU 核心，并且只做
"Read-Resize-Write" 操作，I/O 效率最高。
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
from tqdm import tqdm

# 查找所有支持的图像
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def resize_one_image(args_tuple):
    """
    处理单个图像的工作函数（用于多进程）。
    """
    image_path, images_dir_base, output_dir_base, input_size, interpolation_flag = args_tuple

    try:
        # 1. 计算输出路径
        relative_path = image_path.relative_to(images_dir_base)
        output_path = (output_dir_base / relative_path).with_suffix(".jpg")

        # 2. 如果已存在，则跳过
        if output_path.exists():
            return "skipped"

        # 3. 创建目标目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. 读取 -> Resize -> 写入
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return f"Error: Failed to read {image_path}"

        if img.shape[0] != input_size or img.shape[1] != input_size:
            img = cv2.resize(img, (input_size, input_size), interpolation=interpolation_flag)

        # 5. 保存图像（统一输出为 JPG）
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return "processed"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="并行 Resize 图像")
    parser.add_argument("--images-dir", required=True, type=Path, help="原始图像所在的根目录。")
    parser.add_argument("--output-dir", required=True, type=Path, help="存放 1024x1024 图像的新目录。")
    parser.add_argument("--input-size", type=int, default=1024, help="目标分辨率（默认 1024）。")
    parser.add_argument("--workers", type=int, default=None, help="使用的工作进程数（默认：全部 CPU 核心）。")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=["area", "linear", "cubic"],
        help="Resize 算法: area (慢, 适合缩小), linear (快, 推荐), cubic (中)",
    )
    args = parser.parse_args()

    # 解析插值算法
    if args.interpolation == "area":
        interpolation_flag = cv2.INTER_AREA
    elif args.interpolation == "cubic":
        interpolation_flag = cv2.INTER_CUBIC
    else:
        interpolation_flag = cv2.INTER_LINEAR

    images_dir_base = args.images_dir.resolve()
    output_dir_base = args.output_dir.resolve()

    if not images_dir_base.exists():
        print(f"Error: 图像目录不存在: {images_dir_base}")
        sys.exit(1)

    output_dir_base.mkdir(parents=True, exist_ok=True)
    print(f"原始图像: {images_dir_base}")
    print(f"输出目录: {output_dir_base}")
    print(f"目标尺寸: {args.input_size}x{args.input_size}")
    print(f"Resize算法: {args.interpolation.upper()}")

    print("正在查找所有图像文件...")
    all_image_paths = []
    for ext in SUPPORTED_EXTS:
        all_image_paths.extend(images_dir_base.rglob(f"*{ext}"))

    if not all_image_paths:
        print("未找到任何图像。")
        return

    print(f"总共找到 {len(all_image_paths)} 张图像。")

    num_workers = args.workers if args.workers else cpu_count()
    print(f"使用 {num_workers} 个 CPU 核心进行处理...")

    tasks = [
        (path, images_dir_base, output_dir_base, args.input_size, interpolation_flag)
        for path in all_image_paths
    ]

    processed_count = 0
    skipped_count = 0
    error_count = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Resizing images", unit="file") as pbar:
            for result in pool.imap_unordered(resize_one_image, tasks):
                if result == "processed":
                    processed_count += 1
                elif result == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    if error_count == 1:
                        print(f"\n[ERROR] {result}")
                pbar.update(1)

    print("\n=== 处理完成 ===")
    print(f"成功处理: {processed_count}")
    print(f"已存在跳过: {skipped_count}")
    print(f"处理失败: {error_count}")
    print(f"已缩放图像保存在: {output_dir_base}")


if __name__ == "__main__":
    main()


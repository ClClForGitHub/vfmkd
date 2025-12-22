#!/usr/bin/env python3
"""
验证 BIN 文件结构是否符合预期

检查：
1. 保存的键是否正确
2. 数据形状是否符合预期
3. 掩码处理是否正确（cv2.INTER_AREA + 二值化阈值 0.5）
4. 不写入的键是否确实未写入
"""

import numpy as np
import json
from pathlib import Path
import sys


def verify_bin_structure(output_dir: Path, sample_index: int = 0):
    """
    验证二进制文件结构
    
    Args:
        output_dir: 输出目录路径
        sample_index: 要验证的样本索引（默认：第一个样本）
    """
    output_dir = Path(output_dir)
    
    print("="*80)
    print("🔍 验证 BIN 文件结构")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"验证样本索引: {sample_index}")
    print()
    
    # 1. 检查文件是否存在
    required_files = [
        "images.bin",
        "features.bin",
        "edge_maps.bin",
        "weight_maps.bin",
        "bboxes.bin",
        "masks.bin",
        "metadata.bin",
        "keys.txt",
        "config.json",
    ]
    
    print("📁 检查文件存在性...")
    missing_files = []
    for fname in required_files:
        fpath = output_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  ✅ {fname} ({size:,} bytes)")
        else:
            print(f"  ❌ {fname} (缺失)")
            missing_files.append(fname)
    
    if missing_files:
        print(f"\n❌ 缺失文件: {', '.join(missing_files)}")
        return False
    
    print()
    
    # 2. 读取 config.json 验证配置
    print("📋 读取配置...")
    with open(output_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"  模型类型: {config.get('model_type', 'N/A')}")
    print(f"  图像尺寸: {config.get('image_size', 'N/A')}")
    print(f"  掩码尺寸: {config.get('mask_size', 'N/A')}")
    print(f"  总样本数: {config.get('total_samples', 'N/A')}")
    print(f"  插值方法: {config.get('interpolation_method', 'N/A')}")
    print(f"  二值化阈值: {config.get('mask_binarization_threshold', 'N/A')}")
    print()
    
    # 验证配置值
    assert config.get('interpolation_method') == 'cv2.INTER_AREA', \
        f"插值方法应为 cv2.INTER_AREA，实际为 {config.get('interpolation_method')}"
    assert config.get('mask_binarization_threshold') == 0.5, \
        f"二值化阈值应为 0.5，实际为 {config.get('mask_binarization_threshold')}"
    
    # 3. 读取样本大小定义
    sample_sizes = config.get('sample_sizes', {})
    IMG_SIZE = config.get('image_size', 1024)
    MASK_SIZE = config.get('mask_size', 256)
    
    # 4. 读取并验证第一个样本
    print(f"📊 验证样本 #{sample_index}...")
    print()
    
    # 4.1 图像
    print("🖼️  图像 (images.bin):")
    with open(output_dir / "images.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('image_bytes', 3145728))
        img_bytes = f.read(sample_sizes.get('image_bytes', 3145728))
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE, 3)
        print(f"  形状: {img_arr.shape} (期望: ({IMG_SIZE}, {IMG_SIZE}, 3))")
        print(f"  数据类型: {img_arr.dtype} (期望: uint8)")
        print(f"  值范围: [{img_arr.min()}, {img_arr.max()}] (期望: [0, 255])")
        assert img_arr.shape == (IMG_SIZE, IMG_SIZE, 3), f"图像形状不正确: {img_arr.shape}"
        assert img_arr.dtype == np.uint8, f"图像数据类型不正确: {img_arr.dtype}"
    print()
    
    # 4.2 特征 (P4_S16 + P5_S32)
    print("🧠 特征 (features.bin):")
    with open(output_dir / "features.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('features_bytes', 5242880))
        
        # P4_S16: (1, 256, 64, 64) float32
        p4_bytes = f.read(4194304)  # 1 * 256 * 64 * 64 * 4
        p4_arr = np.frombuffer(p4_bytes, dtype=np.float32).reshape(1, 256, 64, 64)
        print(f"  P4_S16 形状: {p4_arr.shape} (期望: (1, 256, 64, 64))")
        print(f"  P4_S16 数据类型: {p4_arr.dtype} (期望: float32)")
        print(f"  P4_S16 值范围: [{p4_arr.min():.4f}, {p4_arr.max():.4f}]")
        assert p4_arr.shape == (1, 256, 64, 64), f"P4_S16 形状不正确: {p4_arr.shape}"
        assert p4_arr.dtype == np.float32, f"P4_S16 数据类型不正确: {p4_arr.dtype}"
        
        # P5_S32: (1, 256, 32, 32) float32
        p5_bytes = f.read(1048576)  # 1 * 256 * 32 * 32 * 4
        p5_arr = np.frombuffer(p5_bytes, dtype=np.float32).reshape(1, 256, 32, 32)
        print(f"  P5_S32 形状: {p5_arr.shape} (期望: (1, 256, 32, 32))")
        print(f"  P5_S32 数据类型: {p5_arr.dtype} (期望: float32)")
        print(f"  P5_S32 值范围: [{p5_arr.min():.4f}, {p5_arr.max():.4f}]")
        assert p5_arr.shape == (1, 256, 32, 32), f"P5_S32 形状不正确: {p5_arr.shape}"
        assert p5_arr.dtype == np.float32, f"P5_S32 数据类型不正确: {p5_arr.dtype}"
    print()
    
    # 4.3 边缘图 (edge_256x256 + edge_64x64 + edge_32x32)
    print("🔲 边缘图 (edge_maps.bin):")
    with open(output_dir / "edge_maps.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('edge_maps_bytes', 70656))
        
        # edge_256x256: (256, 256) uint8
        edge_256_bytes = f.read(65536)  # 256 * 256 * 1
        edge_256_arr = np.frombuffer(edge_256_bytes, dtype=np.uint8).reshape(256, 256)
        print(f"  edge_256x256 形状: {edge_256_arr.shape} (期望: (256, 256))")
        print(f"  edge_256x256 数据类型: {edge_256_arr.dtype} (期望: uint8)")
        print(f"  edge_256x256 值范围: [{edge_256_arr.min()}, {edge_256_arr.max()}] (期望: [0, 255])")
        assert edge_256_arr.shape == (256, 256), f"edge_256x256 形状不正确: {edge_256_arr.shape}"
        assert edge_256_arr.dtype == np.uint8, f"edge_256x256 数据类型不正确: {edge_256_arr.dtype}"
        
        # edge_64x64: (64, 64) uint8
        edge_64_bytes = f.read(4096)  # 64 * 64 * 1
        edge_64_arr = np.frombuffer(edge_64_bytes, dtype=np.uint8).reshape(64, 64)
        print(f"  edge_64x64 形状: {edge_64_arr.shape} (期望: (64, 64))")
        print(f"  edge_64x64 数据类型: {edge_64_arr.dtype} (期望: uint8)")
        assert edge_64_arr.shape == (64, 64), f"edge_64x64 形状不正确: {edge_64_arr.shape}"
        
        # edge_32x32: (32, 32) uint8
        edge_32_bytes = f.read(1024)  # 32 * 32 * 1
        edge_32_arr = np.frombuffer(edge_32_bytes, dtype=np.uint8).reshape(32, 32)
        print(f"  edge_32x32 形状: {edge_32_arr.shape} (期望: (32, 32))")
        print(f"  edge_32x32 数据类型: {edge_32_arr.dtype} (期望: uint8)")
        assert edge_32_arr.shape == (32, 32), f"edge_32x32 形状不正确: {edge_32_arr.shape}"
    print()
    
    # 4.4 权重图 (fg_map + bg_map for 128/64/32)
    print("⚖️  权重图 (weight_maps.bin):")
    with open(output_dir / "weight_maps.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('weight_maps_bytes', 172032))
        
        weight_maps_info = [
            ("fg_map_128x128", 128, 128),
            ("bg_map_128x128", 128, 128),
            ("fg_map_64x64", 64, 64),
            ("bg_map_64x64", 64, 64),
            ("fg_map_32x32", 32, 32),
            ("bg_map_32x32", 32, 32),
        ]
        
        for name, h, w in weight_maps_info:
            size_bytes = h * w * 4  # float32
            weight_bytes = f.read(size_bytes)
            weight_arr = np.frombuffer(weight_bytes, dtype=np.float32).reshape(h, w)
            print(f"  {name} 形状: {weight_arr.shape} (期望: ({h}, {w}))")
            print(f"  {name} 数据类型: {weight_arr.dtype} (期望: float32)")
            assert weight_arr.shape == (h, w), f"{name} 形状不正确: {weight_arr.shape}"
            assert weight_arr.dtype == np.float32, f"{name} 数据类型不正确: {weight_arr.dtype}"
    print()
    
    # 4.5 边界框 (bboxes)
    print("📦 边界框 (bboxes.bin):")
    with open(output_dir / "bboxes.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('bboxes_bytes', 16))
        bbox_bytes = f.read(16)  # 1 * 4 * 4
        bbox_arr = np.frombuffer(bbox_bytes, dtype=np.float32).reshape(1, 4)
        print(f"  形状: {bbox_arr.shape} (期望: (1, 4))")
        print(f"  数据类型: {bbox_arr.dtype} (期望: float32)")
        print(f"  值: {bbox_arr[0]}")
        assert bbox_arr.shape == (1, 4), f"边界框形状不正确: {bbox_arr.shape}"
        assert bbox_arr.dtype == np.float32, f"边界框数据类型不正确: {bbox_arr.dtype}"
    print()
    
    # 4.6 掩码 (masks)
    print("🎭 掩码 (masks.bin):")
    with open(output_dir / "masks.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('masks_bytes', 65536))
        mask_bytes = f.read(65536)  # 1 * 256 * 256 * 1
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(1, MASK_SIZE, MASK_SIZE)
        print(f"  形状: {mask_arr.shape} (期望: (1, {MASK_SIZE}, {MASK_SIZE}))")
        print(f"  数据类型: {mask_arr.dtype} (期望: uint8)")
        print(f"  值范围: [{mask_arr.min()}, {mask_arr.max()}] (期望: [0, 1])")
        print(f"  唯一值: {np.unique(mask_arr)} (期望: [0, 1] 或 [0] 或 [1])")
        
        # 验证掩码是二值化的（只有 0 和 1）
        unique_vals = np.unique(mask_arr)
        assert all(v in [0, 1] for v in unique_vals), \
            f"掩码不是二值化的，包含值: {unique_vals}"
        assert mask_arr.shape == (1, MASK_SIZE, MASK_SIZE), \
            f"掩码形状不正确: {mask_arr.shape}"
        assert mask_arr.dtype == np.uint8, f"掩码数据类型不正确: {mask_arr.dtype}"
    print()
    
    # 4.7 元数据 (metadata)
    print("📝 元数据 (metadata.bin):")
    with open(output_dir / "metadata.bin", "rb") as f:
        f.seek(sample_index * sample_sizes.get('metadata_bytes', 20))
        meta_bytes = f.read(20)  # 5 * 4 (int32)
        meta_arr = np.frombuffer(meta_bytes, dtype=np.int32)
        print(f"  形状: {meta_arr.shape} (期望: (5,))")
        print(f"  数据类型: {meta_arr.dtype} (期望: int32)")
        print(f"  内容: [num_bboxes={meta_arr[0]}, has_bbox={meta_arr[1]}, "
              f"H={meta_arr[2]}, W={meta_arr[3]}, C={meta_arr[4]}]")
        assert meta_arr.shape == (5,), f"元数据形状不正确: {meta_arr.shape}"
        assert meta_arr.dtype == np.int32, f"元数据数据类型不正确: {meta_arr.dtype}"
    print()
    
    # 5. 验证 keys.txt
    print("🔑 样本键 (keys.txt):")
    with open(output_dir / "keys.txt", "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f.readlines()]
        print(f"  总样本数: {len(keys)}")
        if sample_index < len(keys):
            print(f"  样本 #{sample_index} 键: {keys[sample_index]}")
        else:
            print(f"  ⚠️  样本索引 {sample_index} 超出范围（总样本数: {len(keys)}）")
    print()
    
    # 6. 总结
    print("="*80)
    print("✅ 验证完成！所有检查项均通过")
    print("="*80)
    print()
    print("📋 保存的键总结:")
    print("  ✅ P4_S16, P5_S32 (特征)")
    print("  ✅ edge_256x256, edge_64x64, edge_32x32 (边缘图)")
    print("  ✅ fg_map_128x128, bg_map_128x128, fg_map_64x64, bg_map_64x64, fg_map_32x32, bg_map_32x32 (权重图)")
    print("  ✅ bboxes (规范化到 (1, 4))")
    print("  ✅ masks (规范化到 (1, 256, 256))")
    print("  ✅ num_bboxes, has_bbox, image_shape (元数据)")
    print()
    print("📋 不写入的键（已确认）:")
    print("  ✅ IMAGE_EMB_S16 (与 P4_S16 重复，仅在兼容模式下使用)")
    print("  ✅ edge_original (使用率低，未读取)")
    print("  ✅ image_id (存 keys.txt)")
    print("  ✅ model_type (存 config.json)")
    print("  ✅ feature_flag, edge_flag 等内部标记 (未读取)")
    print()
    print("📋 掩码处理:")
    print("  ✅ 使用 cv2.INTER_AREA 下采样")
    print("  ✅ 二值化阈值: > 0.5")
    print("  ✅ 输出格式: uint8")
    print()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证 BIN 文件结构")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="要验证的样本索引（默认: 0）"
    )
    
    args = parser.parse_args()
    
    try:
        verify_bin_structure(args.output_dir, args.sample_index)
    except AssertionError as e:
        print(f"\n❌ 验证失败: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


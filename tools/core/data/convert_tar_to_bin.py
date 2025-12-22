#!/usr/bin/env python3
"""
TAR → BIN 转换工具（高性能定长二进制存储 + 多进程并行处理）

将 tar 文件中的 JPG 和 NPZ 转换为定长的二进制文件，实现极致 IO 性能。

**性能优化**：
- 使用多进程并行处理（默认使用所有 CPU 核心）
- 主进程负责快速读取 TAR 文件和写入结果（IO 密集）
- Worker 进程负责 CPU 密集操作（JPG 解码、图像缩放、NPZ 加载、数据转换）
- 批处理控制内存积压，避免 OOM
- 预期性能提升：10-20 倍（取决于 CPU 核心数）

输出文件结构:
- images.bin: (1024, 1024, 3) uint8 - 每个样本 3,145,728 bytes
- features.bin: P4_S16 + P5_S32 - 每个样本 5,242,880 bytes
- edge_maps.bin: edge_256x256 + edge_64x64 + edge_32x32 - 每个样本 70,656 bytes
- weight_maps.bin: fg_map_* + bg_map_* - 每个样本 172,032 bytes
- bboxes.bin: (1, 4) float32 - 每个样本 16 bytes
- masks.bin: (1, 256, 256) uint8 - 每个样本 65,536 bytes
- metadata.bin: num_bboxes + has_bbox + image_shape - 每个样本 20 bytes
- keys.txt: 样本 ID 列表（用于调试）
- config.json: 全局配置（model_type 等）

写入的键:
- P4_S16, P5_S32 (特征)
- edge_256x256, edge_64x64, edge_32x32 (边缘图)
- fg_map_128x128, bg_map_128x128, fg_map_64x64, bg_map_64x64, fg_map_32x32, bg_map_32x32 (权重图)
- bboxes (规范化), masks (规范化)
- num_bboxes, has_bbox, image_shape (元数据)

不写入的键:
- IMAGE_EMB_S16 (与 P4_S16 重复)
- edge_original (使用率低，仅9%)
- image_id (存 keys.txt)
- model_type (存 config.json)
- feature_flag, edge_flag, edge_version, geometry_color_flag (内部标记，不需要)

用法:
    python convert_tar_to_bin.py \
        --tar-path /path/to/sa1b_shard_00000.tar \
        --output-dir ./binary_dataset \
        [--max-samples 1000] \
        [--model-type "sam2.1_hiera_b+"] \
        [--workers 32]  # 并行 Worker 进程数（默认：所有 CPU 核心）
"""

import argparse
import tarfile
import numpy as np
import cv2
import io
import json
import os
import sys
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple, List
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ==================== 配置常量 ====================

# 尺寸定义
IMG_SIZE = 1024
MASK_SIZE = 256

# 每个样本的字节大小（用于验证）
SAMPLE_SIZE_IMAGE = IMG_SIZE * IMG_SIZE * 3  # 3,145,728 bytes
SAMPLE_SIZE_P4 = 1 * 256 * 64 * 64 * 4  # 4,194,304 bytes (float32)
SAMPLE_SIZE_P5 = 1 * 256 * 32 * 32 * 4  # 1,048,576 bytes (float32)
SAMPLE_SIZE_FEATURES = SAMPLE_SIZE_P4 + SAMPLE_SIZE_P5  # 5,242,880 bytes
SAMPLE_SIZE_EDGE_256 = 256 * 256 * 1  # 65,536 bytes (uint8)
SAMPLE_SIZE_EDGE_64 = 64 * 64 * 1  # 4,096 bytes (uint8)
SAMPLE_SIZE_EDGE_32 = 32 * 32 * 1  # 1,024 bytes (uint8)
SAMPLE_SIZE_EDGE_MAPS = SAMPLE_SIZE_EDGE_256 + SAMPLE_SIZE_EDGE_64 + SAMPLE_SIZE_EDGE_32  # 70,656 bytes
SAMPLE_SIZE_WEIGHT_128 = 128 * 128 * 4  # 65,536 bytes (float32)
SAMPLE_SIZE_WEIGHT_64 = 64 * 64 * 4  # 16,384 bytes (float32)
SAMPLE_SIZE_WEIGHT_32 = 32 * 32 * 4  # 4,096 bytes (float32)
SAMPLE_SIZE_WEIGHT_MAPS = (SAMPLE_SIZE_WEIGHT_128 + SAMPLE_SIZE_WEIGHT_64 + SAMPLE_SIZE_WEIGHT_32) * 2  # 172,032 bytes
SAMPLE_SIZE_BBOX = 1 * 4 * 4  # 16 bytes (float32)
SAMPLE_SIZE_MASK = 1 * MASK_SIZE * MASK_SIZE * 1  # 65,536 bytes (uint8)
SAMPLE_SIZE_METADATA = 5 * 4  # 20 bytes (5个 int32)


# ==================== 辅助函数 ====================

def normalize_mask(mask_arr: np.ndarray, target_size: int = MASK_SIZE) -> np.ndarray:
    """
    将变长掩码规范化到固定尺寸 256x256
    
    使用 cv2.INTER_AREA 插值方法（与特征提取和训练脚本一致）
    
    Args:
        mask_arr: 原始掩码，可能是 (H, W) 或 (1, H, W) 或 object 数组
        target_size: 目标尺寸，默认 256
    
    Returns:
        规范化后的掩码，形状 (1, 256, 256)，dtype=uint8
    """
    # 确保是 numpy 数组（兼容标量、列表等）
    if not isinstance(mask_arr, np.ndarray):
        # 如果是标量或其他类型，返回全零掩码
        if isinstance(mask_arr, (int, float)) or mask_arr is None:
            return np.zeros((1, target_size, target_size), dtype=np.uint8)
        mask_arr = np.array(mask_arr)
    
    # 处理 object 数组（masks 可能是 object 类型）
    if mask_arr.dtype == object:
        if mask_arr.size == 0:
            # 空掩码，返回全零
            return np.zeros((1, target_size, target_size), dtype=np.uint8)
        # 取第一个元素
        mask_item = mask_arr.item(0)
        # 确保取出的元素是 numpy 数组
        if not isinstance(mask_item, np.ndarray):
            mask_arr = np.array(mask_item)
        else:
            mask_arr = mask_item
    
    # 再次确保是 numpy 数组
    if not isinstance(mask_arr, np.ndarray):
        mask_arr = np.array(mask_arr)
    
    # 确保是 2D 或 3D
    if mask_arr.ndim == 0:
        # 0 维数组（标量），返回全零掩码
        return np.zeros((1, target_size, target_size), dtype=np.uint8)
    elif mask_arr.ndim == 1:
        # 1 维数组，可能是不规则形状，返回全零掩码
        return np.zeros((1, target_size, target_size), dtype=np.uint8)
    elif mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]  # (H, W) -> (1, H, W)
    elif mask_arr.ndim == 3 and mask_arr.shape[0] > 1:
        # 如果多个通道，取第一个
        mask_arr = mask_arr[0:1, ...]
    elif mask_arr.ndim > 3:
        # 超过 3 维，尝试降维
        while mask_arr.ndim > 3:
            mask_arr = mask_arr[0]
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
    
    # 验证维度
    if mask_arr.ndim != 3:
        # 无法处理的维度，返回全零掩码
        return np.zeros((1, target_size, target_size), dtype=np.uint8)
    
    # 提取 H, W
    if mask_arr.shape[0] == 0:
        return np.zeros((1, target_size, target_size), dtype=np.uint8)
    _, h, w = mask_arr.shape
    
    # 使用 cv2.INTER_AREA 下采样（与特征提取和训练脚本一致）
    # 参考: tools/core/bbox/sa1b_bbox_extractor.py:392
    # 参考: tools/core/exper/train_distill_single_test.py:633
    mask_2d = mask_arr[0].astype(np.float32)
    mask_resized = cv2.resize(
        mask_2d,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )
    
    # 二值化：> 0.5 则为 1（与特征提取脚本一致）
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # 添加 batch 维度
    return mask_binary[None, ...]  # (1, 256, 256)


def normalize_bbox(bbox_arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    规范化边界框到固定形状 (1, 4)
    
    Args:
        bbox_arr: 原始边界框，可能是 (0, 4) 或 (1, 4) 或标量/列表
    
    Returns:
        (fixed_bbox, num_bboxes, has_bbox)
        - fixed_bbox: (1, 4) float32
        - num_bboxes: 实际框数量
        - has_bbox: 是否有框 (0/1)
    """
    # 确保是 numpy 数组
    if not isinstance(bbox_arr, np.ndarray):
        if bbox_arr is None or (isinstance(bbox_arr, (int, float)) and bbox_arr == 0):
            bbox_arr = np.empty((0, 4), dtype=np.float32)
        else:
            bbox_arr = np.array(bbox_arr, dtype=np.float32)
    
    # 确保是 2D
    if bbox_arr.ndim == 1:
        bbox_arr = bbox_arr[None, ...]  # (4,) -> (1, 4)
    elif bbox_arr.ndim == 0:
        # 标量，当作空框处理
        bbox_arr = np.empty((0, 4), dtype=np.float32)
    
    if bbox_arr.shape[0] == 0:
        # 无框时：填充左上角小框 [0, 0, 1, 1]（与训练代码一致）
        # 参考: tools/core/exper/train_distill_single_test.py:2079
        fixed_bbox = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        num_bboxes = 0
        has_bbox = 0
    else:
        # 有框时：直接使用（确保是 (1, 4)）
        fixed_bbox = bbox_arr.astype(np.float32)
        if fixed_bbox.ndim == 1:
            fixed_bbox = fixed_bbox[None, ...]  # (4,) -> (1, 4)
        num_bboxes = fixed_bbox.shape[0]
        has_bbox = 1
    
    # 确保形状是 (1, 4)
    if fixed_bbox.shape[0] > 1:
        fixed_bbox = fixed_bbox[0:1, ...]  # 只取第一个框
    
    return fixed_bbox, num_bboxes, has_bbox


def jpg_to_raw(jpg_bytes: bytes, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    将 JPG 解码为固定大小的 RGB 数组
    
    Args:
        jpg_bytes: JPG 文件的字节数据
        target_size: 目标尺寸，默认 1024
    
    Returns:
        RGB 图像数组，形状 (1024, 1024, 3)，dtype=uint8
    """
    # 解码 JPG
    img_arr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_arr is None:
        raise ValueError("无法解码 JPG 图像")
    
    # 转换为 RGB
    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    
    # 确保尺寸严格为 target_size x target_size
    if img_rgb.shape[:2] != (target_size, target_size):
        img_rgb = cv2.resize(
            img_rgb,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 确保是 uint8
    return img_rgb.astype(np.uint8)  # (1024, 1024, 3)


# ==================== Worker 函数（多进程处理） ====================

def process_single_pair(args: Tuple[str, bytes, bytes, int, int]) -> Optional[Dict]:
    """
    Worker 进程执行的函数：接收原始字节，返回处理好的定长字节数据
    
    这是一个顶级函数（必须在模块级别），以便 multiprocessing 可以 pickle 它。
    
    Args:
        args: (base_name, jpg_bytes, npz_bytes, img_size, mask_size)
    
    Returns:
        处理结果字典，包含所有需要写入的二进制数据，如果处理失败返回 None
    """
    base_name, jpg_bytes, npz_bytes, img_size, mask_size = args
    
    try:
        # === 1. 处理图像 (CPU 密集：JPG 解码 + Resize) ===
        img_arr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_arr is None:
            raise ValueError("无法解码 JPG 图像")
        
        # 转换为 RGB
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        
        # 确保尺寸严格为 img_size x img_size
        if img_rgb.shape[:2] != (img_size, img_size):
            img_rgb = cv2.resize(
                img_rgb,
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
        
        img_raw = img_rgb.astype(np.uint8)
        
        # 验证尺寸
        if img_raw.shape != (img_size, img_size, 3):
            raise ValueError(f"图像尺寸不正确: {img_raw.shape}, 期望: ({img_size}, {img_size}, 3)")
        
        # === 2. 处理 NPZ 数据 (CPU 密集：NPZ 加载 + 数据转换) ===
        npz_buffer = io.BytesIO(npz_bytes)
        with np.load(npz_buffer, allow_pickle=True) as data:
            # A. Features (P4_S16 + P5_S32)
            if "P4_S16" in data:
                p4 = data["P4_S16"]
            elif "IMAGE_EMB_S16" in data:
                # 如果只有 IMAGE_EMB_S16，使用它（兼容旧格式）
                p4 = data["IMAGE_EMB_S16"]
            else:
                raise KeyError("未找到 P4_S16 或 IMAGE_EMB_S16")
            
            if "P5_S32" not in data:
                raise KeyError("未找到 P5_S32")
            p5 = data["P5_S32"]
            
            # 确保是 numpy 数组并转换类型
            if not isinstance(p4, np.ndarray):
                p4 = np.array(p4, dtype=np.float32)
            else:
                p4 = p4.astype(np.float32)
            
            if not isinstance(p5, np.ndarray):
                p5 = np.array(p5, dtype=np.float32)
            else:
                p5 = p5.astype(np.float32)
            
            # 确保有 batch 维度（再次检查是否为数组）
            if not isinstance(p4, np.ndarray):
                raise ValueError(f"P4_S16 不是 numpy 数组: {type(p4)}")
            if not isinstance(p5, np.ndarray):
                raise ValueError(f"P5_S32 不是 numpy 数组: {type(p5)}")
            
            if p4.ndim == 3:
                p4 = p4[None, ...]
            if p5.ndim == 3:
                p5 = p5[None, ...]
            
            # 验证形状
            if p4.shape != (1, 256, 64, 64):
                raise ValueError(f"P4_S16 形状不正确: {p4.shape}, 期望: (1, 256, 64, 64)")
            if p5.shape != (1, 256, 32, 32):
                raise ValueError(f"P5_S32 形状不正确: {p5.shape}, 期望: (1, 256, 32, 32)")
            
            # B. Edge Maps
            # 确保字段存在并是 numpy 数组
            for key in ["edge_256x256", "edge_64x64", "edge_32x32"]:
                if key not in data:
                    raise KeyError(f"缺失字段: {key}")
            
            raw_edge_256 = data["edge_256x256"]
            raw_edge_64 = data["edge_64x64"]
            raw_edge_32 = data["edge_32x32"]
            
            # 处理 0 维数组（标量数组）
            if isinstance(raw_edge_256, np.ndarray) and raw_edge_256.ndim == 0:
                raise ValueError(f"edge_256x256 是标量数组，期望 2D 数组")
            if isinstance(raw_edge_64, np.ndarray) and raw_edge_64.ndim == 0:
                raise ValueError(f"edge_64x64 是标量数组，期望 2D 数组")
            if isinstance(raw_edge_32, np.ndarray) and raw_edge_32.ndim == 0:
                raise ValueError(f"edge_32x32 是标量数组，期望 2D 数组")
            
            if not isinstance(raw_edge_256, np.ndarray):
                raw_edge_256 = np.array(raw_edge_256, dtype=np.uint8)
            if not isinstance(raw_edge_64, np.ndarray):
                raw_edge_64 = np.array(raw_edge_64, dtype=np.uint8)
            if not isinstance(raw_edge_32, np.ndarray):
                raw_edge_32 = np.array(raw_edge_32, dtype=np.uint8)
            
            edge_256 = raw_edge_256.astype(np.uint8)
            edge_64 = raw_edge_64.astype(np.uint8)
            edge_32 = raw_edge_32.astype(np.uint8)
            
            # 验证形状
            if edge_256.shape != (256, 256):
                raise ValueError(f"edge_256x256 形状不正确: {edge_256.shape}, 期望: (256, 256)")
            if edge_64.shape != (64, 64):
                raise ValueError(f"edge_64x64 形状不正确: {edge_64.shape}, 期望: (64, 64)")
            if edge_32.shape != (32, 32):
                raise ValueError(f"edge_32x32 形状不正确: {edge_32.shape}, 期望: (32, 32)")
            
            # C. Weight Maps
            # 确保字段存在并是 numpy 数组
            weight_map_keys = [
                "fg_map_128x128", "bg_map_128x128",
                "fg_map_64x64", "bg_map_64x64",
                "fg_map_32x32", "bg_map_32x32",
            ]
            for key in weight_map_keys:
                if key not in data:
                    raise KeyError(f"缺失字段: {key}")
            
            weight_maps_data = {
                key: data[key] for key in weight_map_keys
            }
            
            weight_maps = {}
            for name, raw_arr in weight_maps_data.items():
                # 检查是否为标量数组
                if isinstance(raw_arr, np.ndarray) and raw_arr.ndim == 0:
                    raise ValueError(f"{name} 是标量数组，期望 2D 数组")
                if not isinstance(raw_arr, np.ndarray):
                    raw_arr = np.array(raw_arr, dtype=np.float32)
                weight_maps[name] = raw_arr.astype(np.float32)
            
            fg_128 = weight_maps["fg_map_128x128"]
            bg_128 = weight_maps["bg_map_128x128"]
            fg_64 = weight_maps["fg_map_64x64"]
            bg_64 = weight_maps["bg_map_64x64"]
            fg_32 = weight_maps["fg_map_32x32"]
            bg_32 = weight_maps["bg_map_32x32"]
            
            # 验证形状
            for name, arr in [
                ("fg_map_128x128", fg_128),
                ("bg_map_128x128", bg_128),
                ("fg_map_64x64", fg_64),
                ("bg_map_64x64", bg_64),
                ("fg_map_32x32", fg_32),
                ("bg_map_32x32", bg_32),
            ]:
                expected_shape = tuple(map(int, name.split('_')[-1].split('x')))
                if arr.shape != expected_shape:
                    raise ValueError(f"{name} 形状不正确: {arr.shape}, 期望: {expected_shape}")
            
            # D. BBoxes (规范化)
            if "bboxes" in data:
                raw_bbox = data["bboxes"]
            else:
                raw_bbox = np.empty((0, 4), dtype=np.float32)
            
            fixed_bbox, num_bboxes, has_bbox = normalize_bbox(raw_bbox)
            
            # E. Masks (规范化)
            if "masks" in data:
                raw_mask = data["masks"]
                fixed_mask = normalize_mask(raw_mask, target_size=mask_size)
            else:
                fixed_mask = np.zeros((1, mask_size, mask_size), dtype=np.uint8)
            
            # 验证掩码形状
            if fixed_mask.shape != (1, mask_size, mask_size):
                raise ValueError(f"规范化后的掩码形状不正确: {fixed_mask.shape}, 期望: (1, {mask_size}, {mask_size})")
            
            # F. Metadata
            if "image_shape" in data:
                raw_shape = data["image_shape"]
                # 先转换为 numpy 数组（兼容标量、列表、数组等）
                if not isinstance(raw_shape, np.ndarray):
                    raw_shape = np.array(raw_shape, dtype=np.int32)
                else:
                    raw_shape = raw_shape.astype(np.int32)
                
                # 处理不同形状
                if raw_shape.ndim == 0:
                    # 标量，使用默认形状
                    image_shape = np.array([img_size, img_size, 3], dtype=np.int32)
                elif raw_shape.size == 3:
                    # 有3个元素，重塑为 (3,)
                    image_shape = raw_shape.reshape(3) if raw_shape.ndim > 1 else raw_shape.flatten()
                else:
                    raise ValueError(f"image_shape 形状不正确: {raw_shape.shape}, 期望: (3,)")
            else:
                image_shape = np.array([img_size, img_size, 3], dtype=np.int32)
            
            # 元数据: [num_bboxes, has_bbox, H, W, C] (5个 int32)
            meta_vec = np.array(
                [num_bboxes, has_bbox, image_shape[0], image_shape[1], image_shape[2]],
                dtype=np.int32
            )
        
        # === 3. 返回所有处理好的二进制数据 ===
        return {
            "name": base_name,
            "img": img_raw.tobytes(),
            "feat_p4": p4.tobytes(),
            "feat_p5": p5.tobytes(),
            "edge_256": edge_256.tobytes(),
            "edge_64": edge_64.tobytes(),
            "edge_32": edge_32.tobytes(),
            "fg_128": fg_128.tobytes(),
            "bg_128": bg_128.tobytes(),
            "fg_64": fg_64.tobytes(),
            "bg_64": bg_64.tobytes(),
            "fg_32": fg_32.tobytes(),
            "bg_32": bg_32.tobytes(),
            "bbox": fixed_bbox.tobytes(),
            "mask": fixed_mask.tobytes(),
            "meta": meta_vec.tobytes(),
        }
    
    except Exception as e:
        # 返回错误信息（包含详细堆栈跟踪，方便调试）
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        # 只在开发模式下包含完整堆栈
        if len(str(e)) < 200:  # 如果错误信息简短，可能缺少上下文
            error_detail += f"\n{traceback.format_exc()}"
        return {
            "name": base_name,
            "error": error_detail,
        }


# ==================== 主转换函数 ====================

def convert_tar_to_bin(
    tar_path: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    model_type: str = "sam2.1_hiera_b+",
    verbose: bool = True,
    max_workers: Optional[int] = None,
    append: bool = False,
) -> Dict[str, int]:
    """
    将 TAR 文件转换为定长二进制文件（多进程并行版本）
    
    Args:
        tar_path: TAR 文件路径
        output_dir: 输出目录
        max_samples: 最大转换样本数
        model_type: 模型类型（写入 config.json）
        verbose: 是否显示进度条
        max_workers: 并行 Worker 数量（None 表示使用所有 CPU 核心）
        append: 是否追加模式（True=追加，False=覆盖）
    
    Returns:
        统计信息字典
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置并行度
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # 决定文件打开模式
    mode_bin = "ab" if append else "wb"
    mode_txt = "a" if append else "w"
    
    if verbose:
        mode_str = "追加" if append else "覆盖"
        print(f"🚀 启动并行处理，使用 {max_workers} 个 Worker 进程... (模式: {mode_str})")
    
    # 打开输出文件（根据 append 参数决定模式）
    files = {
        "images": open(output_dir / "images.bin", mode_bin),
        "features": open(output_dir / "features.bin", mode_bin),
        "edge_maps": open(output_dir / "edge_maps.bin", mode_bin),
        "weight_maps": open(output_dir / "weight_maps.bin", mode_bin),
        "bboxes": open(output_dir / "bboxes.bin", mode_bin),
        "masks": open(output_dir / "masks.bin", mode_bin),
        "metadata": open(output_dir / "metadata.bin", mode_bin),
    }
    f_keys = open(output_dir / "keys.txt", mode_txt, encoding="utf-8")
    
    # 统计信息
    stats = {
        "total": 0,
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": [],
    }
    
    # 用于缓存成对的 jpg 和 npz（主进程快速读取）
    buffer: Dict[str, Dict[str, bytes]] = {}
    
    # 批处理配置：控制内存中的任务积压数量
    BATCH_SIZE = max_workers * 4
    
    try:
        with tarfile.open(tar_path, "r|*") as tar:
            # 创建进程池
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 待处理的任务列表（Future 对象）
                pending_futures: List[concurrent.futures.Future] = []
                
                # 进度条（用于显示读取进度）
                if verbose and tqdm:
                    read_pbar = tqdm(desc="读取 TAR", unit="文件")
                else:
                    read_pbar = None
                
                # 第一阶段：快速读取 TAR 文件，提交任务到进程池（生产者）
                for member in tar:
                    if read_pbar is not None:
                        read_pbar.update(1)
                    
                    if not member.isfile():
                        continue
                    
                    # 提取文件名和扩展名
                    name = member.name
                    base_name = os.path.splitext(os.path.basename(name))[0]
                    ext = os.path.splitext(name)[1].lower()
                    
                    # 只处理 .jpg 和 .npz 文件
                    if ext not in ['.jpg', '.npz']:
                        continue
                    
                    # 标准化 base_name（移除 _features 后缀）
                    if base_name.endswith('_features'):
                        base_name = base_name[:-9]  # 移除 '_features'
                    elif base_name.endswith('_sam2_features'):
                        base_name = base_name[:-15]  # 移除 '_sam2_features'
                    
                    # 读取文件内容（只读取原始 bytes，不进行解码）
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read()
                    f.close()
                    
                    # 缓存文件
                    if base_name not in buffer:
                        buffer[base_name] = {}
                    buffer[base_name][ext] = content
                    
                    # 如果一对文件都齐了 (.jpg 和 .npz)
                    if '.jpg' in buffer[base_name] and '.npz' in buffer[base_name]:
                        stats["total"] += 1
                        
                        # 检查是否达到最大样本数
                        if max_samples is not None and stats["success"] >= max_samples:
                            del buffer[base_name]
                            continue
                        
                        # 提交任务到进程池（传递原始 bytes）
                        jpg_bytes = buffer[base_name]['.jpg']
                        npz_bytes = buffer[base_name]['.npz']
                        
                        future = executor.submit(
                            process_single_pair,
                            (base_name, jpg_bytes, npz_bytes, IMG_SIZE, MASK_SIZE)
                        )
                        pending_futures.append(future)
                        
                        # 清理 buffer（已提交处理）
                        del buffer[base_name]
                        
                        # 内存控制：如果积压任务太多，先处理完一批再继续读 TAR
                        if len(pending_futures) >= BATCH_SIZE:
                            # 处理一批任务（消费者）
                            _process_batch_results(
                                pending_futures, files, f_keys, stats, verbose, read_pbar
                            )
                            pending_futures.clear()
                
                # 关闭读取进度条
                if read_pbar is not None:
                    read_pbar.close()
                
                # 第二阶段：处理剩余的任务
                if pending_futures:
                    if verbose and tqdm:
                        process_pbar = tqdm(
                            desc="处理中",
                            total=len(pending_futures),
                            unit="样本"
                        )
                    else:
                        process_pbar = None
                    
                    _process_batch_results(
                        pending_futures, files, f_keys, stats, verbose, process_pbar
                    )
                    
                    if process_pbar is not None:
                        process_pbar.close()
    
    finally:
        # 关闭所有文件
        for f in files.values():
            f.close()
        f_keys.close()
    
    # 生成/更新 config.json
    # model_type 存到 config.json（不写入二进制）
    config_path = output_dir / "config.json"
    
    # 计算总样本数（追加模式需要累加）
    total_samples = stats["success"]
    if append and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                old_config = json.load(f)
                # 累加总样本数
                if "total_samples" in old_config:
                    old_total = int(old_config["total_samples"])
                    total_samples = old_total + stats["success"]
                    if verbose:
                        print(f"📊 累加样本数: {old_total} (已有) + {stats['success']} (本次) = {total_samples} (总计)")
        except Exception as e:
            if verbose:
                print(f"⚠️  警告: 读取旧 config.json 失败: {e}，将使用本次样本数")
    
    config = {
        "model_type": model_type,
        "image_size": IMG_SIZE,
        "mask_size": MASK_SIZE,
        "total_samples": total_samples,  # 使用累加后的值
        "version": "1.0",
        "description": "SA-1B dataset converted to fixed-length binary format",
        "sample_sizes": {
            "image_bytes": SAMPLE_SIZE_IMAGE,
            "features_bytes": SAMPLE_SIZE_FEATURES,
            "edge_maps_bytes": SAMPLE_SIZE_EDGE_MAPS,
            "weight_maps_bytes": SAMPLE_SIZE_WEIGHT_MAPS,
            "bboxes_bytes": SAMPLE_SIZE_BBOX,
            "masks_bytes": SAMPLE_SIZE_MASK,
            "metadata_bytes": SAMPLE_SIZE_METADATA,
        },
        "interpolation_method": "cv2.INTER_AREA",
        "mask_binarization_threshold": 0.5,
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return stats


def _process_batch_results(
    futures: List[concurrent.futures.Future],
    files: Dict[str, any],
    f_keys: any,
    stats: Dict[str, int],
    verbose: bool,
    pbar: Optional[tqdm] = None,
) -> None:
    """
    处理一批任务的结果并写入文件（辅助函数）
    
    Args:
        futures: Future 对象列表
        files: 输出文件句柄字典
        f_keys: keys.txt 文件句柄
        stats: 统计信息字典
        verbose: 是否显示详细信息
        pbar: 进度条对象（可选）
    """
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        
        if result is None:
            stats["errors"] += 1
            continue
        
        # 检查是否有错误
        if "error" in result:
            stats["errors"] += 1
            error_msg = f"{result['name']}: {result['error']}"
            stats["error_details"].append(error_msg)
            if verbose:
                print(f"\n[错误] {error_msg}", file=sys.stderr)
            continue
        
        # 写入所有处理好的二进制数据
        try:
            files["images"].write(result["img"])
            files["features"].write(result["feat_p4"])
            files["features"].write(result["feat_p5"])
            files["edge_maps"].write(result["edge_256"])
            files["edge_maps"].write(result["edge_64"])
            files["edge_maps"].write(result["edge_32"])
            files["weight_maps"].write(result["fg_128"])
            files["weight_maps"].write(result["bg_128"])
            files["weight_maps"].write(result["fg_64"])
            files["weight_maps"].write(result["bg_64"])
            files["weight_maps"].write(result["fg_32"])
            files["weight_maps"].write(result["bg_32"])
            files["bboxes"].write(result["bbox"])
            files["masks"].write(result["mask"])
            files["metadata"].write(result["meta"])
            
            # 记录 Key
            f_keys.write(result["name"] + "\n")
            f_keys.flush()
            
            stats["success"] += 1
            
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "成功": stats["success"],
                    "错误": stats["errors"]
                })
        
        except Exception as e:
            stats["errors"] += 1
            error_msg = f"{result['name']}: 写入失败 - {str(e)}"
            stats["error_details"].append(error_msg)
            if verbose:
                print(f"\n[错误] {error_msg}", file=sys.stderr)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="将 TAR 文件转换为定长二进制文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个 shard（使用所有 CPU 核心）
  python convert_tar_to_bin.py \\
      --tar-path /path/to/sa1b_shard_00000.tar \\
      --output-dir ./binary_dataset \\
      --model-type "sam2.1_hiera_b+"
  
  # 限制转换样本数（测试用）
  python convert_tar_to_bin.py \\
      --tar-path /path/to/sa1b_shard_00000.tar \\
      --output-dir ./binary_dataset \\
      --max-samples 100
  
  # 指定 Worker 进程数（例如：32 核服务器使用 30 个进程）
  python convert_tar_to_bin.py \\
      --tar-path /path/to/sa1b_shard_00000.tar \\
      --output-dir ./binary_dataset \\
      --workers 30
        """
    )
    
    parser.add_argument(
        "--tar-path",
        type=Path,
        required=True,
        help="输入的 TAR 文件路径"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（将创建 .bin 文件）"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大转换样本数（用于测试，默认：全部）"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="sam2.1_hiera_b+",
        help="模型类型（将写入 config.json）"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（不显示进度条）"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"并行 Worker 进程数量（默认：使用所有 CPU 核心，当前为 {multiprocessing.cpu_count()}）"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加模式（不覆盖现有文件，用于批量转换多个 shard 时从第二个开始使用）"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not args.tar_path.exists():
        print(f"❌ 错误：TAR 文件不存在: {args.tar_path}", file=sys.stderr)
        sys.exit(1)
    
    # 执行转换
    print(f"📦 开始转换: {args.tar_path}")
    print(f"📁 输出目录: {args.output_dir}")
    if args.max_samples:
        print(f"🔢 最大样本数: {args.max_samples}")
    if args.workers:
        print(f"👷 Worker 进程数: {args.workers}")
    print()
    
    stats = convert_tar_to_bin(
        tar_path=args.tar_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        model_type=args.model_type,
        verbose=not args.quiet,
        max_workers=args.workers,
        append=args.append,
    )
    
    # 打印统计信息
    print("\n" + "="*80)
    print("📊 转换完成！")
    print("="*80)
    print(f"总样本数: {stats['total']}")
    print(f"成功转换: {stats['success']}")
    print(f"跳过: {stats['skipped']}")
    print(f"错误: {stats['errors']}")
    
    if stats['error_details'] and len(stats['error_details']) <= 10:
        print("\n错误详情:")
        for err in stats['error_details']:
            print(f"  - {err}")
    elif stats['error_details']:
        print(f"\n错误详情（前10个）:")
        for err in stats['error_details'][:10]:
            print(f"  - {err}")
        print(f"  ... 还有 {len(stats['error_details']) - 10} 个错误")
    
    print()
    print(f"✅ 输出文件已保存到: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()


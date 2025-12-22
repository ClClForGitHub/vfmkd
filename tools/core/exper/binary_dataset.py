#!/usr/bin/env python3
"""
高性能定长二进制数据集读取器

对应 convert_tar_to_bin.py 生成的数据格式，实现极速 IO 性能。

设计理念：
1. O(1) Seek: 直接通过 index * ITEM_SIZE 计算偏移量
2. Zero-Copy: 使用 np.frombuffer 直接转换，避免内存拷贝
3. Lazy Initialization: 文件句柄在 __getitem__ 中首次调用时才打开
4. Tensor-Ready: 读取后直接转为 Tensor，包含归一化处理
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple


class BinaryDistillDataset(Dataset):
    """
    高性能定长二进制数据集读取器
    
    对应 convert_tar_to_bin.py 生成的数据格式：
    - images.bin: (1024, 1024, 3) uint8
    - features.bin: P4_S16 + P5_S32 float32
    - edge_maps.bin: edge_256x256 + edge_64x64 + edge_32x32 uint8
    - weight_maps.bin: fg_map_* + bg_map_* float32
    - bboxes.bin: (1, 4) float32
    - masks.bin: (1, 256, 256) uint8
    - metadata.bin: [num_bboxes, has_bbox, H, W, C] int32
    """
    
    # 硬编码尺寸常量（必须与 convert_tar_to_bin.py 一致）
    IMG_SIZE = 1024
    MASK_SIZE = 256
    
    # 字节大小定义
    SIZE_IMG = IMG_SIZE * IMG_SIZE * 3  # 3,145,728 bytes
    SIZE_P4 = 1 * 256 * 64 * 64 * 4  # 4,194,304 bytes (float32)
    SIZE_P5 = 1 * 256 * 32 * 32 * 4  # 1,048,576 bytes (float32)
    SIZE_FEAT = SIZE_P4 + SIZE_P5  # 5,242,880 bytes
    
    # 边缘图大小
    SIZE_EDGE_256 = 256 * 256 * 1  # 65,536 bytes (uint8)
    SIZE_EDGE_64 = 64 * 64 * 1  # 4,096 bytes (uint8)
    SIZE_EDGE_32 = 32 * 32 * 1  # 1,024 bytes (uint8)
    SIZE_EDGE_ALL = SIZE_EDGE_256 + SIZE_EDGE_64 + SIZE_EDGE_32  # 70,656 bytes
    
    # 权重图大小 (6个图: fg_128, bg_128, fg_64, bg_64, fg_32, bg_32)
    SIZE_WEIGHT_128 = 128 * 128 * 4  # 65,536 bytes (float32)
    SIZE_WEIGHT_64 = 64 * 64 * 4  # 16,384 bytes (float32)
    SIZE_WEIGHT_32 = 32 * 32 * 4  # 4,096 bytes (float32)
    SIZE_WEIGHT_ALL = (SIZE_WEIGHT_128 + SIZE_WEIGHT_64 + SIZE_WEIGHT_32) * 2  # 172,032 bytes
    
    SIZE_BBOX = 1 * 4 * 4  # 16 bytes (float32)
    SIZE_MASK = 1 * MASK_SIZE * MASK_SIZE * 1  # 65,536 bytes (uint8)
    SIZE_META = 5 * 4  # 20 bytes (5个 int32)
    
    def __init__(
        self,
        data_root: str,
        input_size: int = 1024,
        verbose: bool = True,
    ):
        """
        Args:
            data_root: 二进制数据集根目录（包含 config.json 和所有 .bin 文件）
            input_size: 输入图像尺寸（默认 1024，必须与转换脚本一致）
            verbose: 是否打印详细信息
        """
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.verbose = verbose
        
        # 读取配置，获取样本总数
        config_path = self.data_root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}. "
                "Please run convert_tar_to_bin.py first."
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.num_samples = self.config.get('total_samples', 0)
        if self.num_samples == 0:
            raise ValueError(f"Config reports 0 samples. Check {config_path}")
        
        # 验证文件存在
        required_files = [
            "images.bin",
            "features.bin",
            "edge_maps.bin",
            "weight_maps.bin",
            "bboxes.bin",
            "masks.bin",
            "metadata.bin",
        ]
        for fname in required_files:
            fpath = self.data_root / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Required file not found: {fpath}")
        
        # 读取 keys.txt（用于调试，可选）
        keys_path = self.data_root / "keys.txt"
        self.image_ids = None
        if keys_path.exists():
            with open(keys_path, 'r', encoding='utf-8') as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
            if len(self.image_ids) != self.num_samples:
                if self.verbose:
                    print(f"Warning: keys.txt has {len(self.image_ids)} lines, "
                          f"but config reports {self.num_samples} samples")
        
        # 文件句柄（惰性初始化，每个 Worker 进程独立打开）
        self.files: Dict[str, Optional] = {}
        
        if self.verbose:
            print(f"[BinaryDataset] 加载二进制数据集: {data_root}")
            print(f"  样本总数: {self.num_samples:,}")
            print(f"  图像尺寸: {self.IMG_SIZE}x{self.IMG_SIZE}")
            print(f"  掩码尺寸: {self.MASK_SIZE}x{self.MASK_SIZE}")
    
    def _init_files(self):
        """每个 Worker 进程独立打开文件句柄（惰性初始化）"""
        if not self.files:
            self.files = {
                "images": open(self.data_root / "images.bin", "rb"),
                "features": open(self.data_root / "features.bin", "rb"),
                "edge_maps": open(self.data_root / "edge_maps.bin", "rb"),
                "weight_maps": open(self.data_root / "weight_maps.bin", "rb"),
                "bboxes": open(self.data_root / "bboxes.bin", "rb"),
                "masks": open(self.data_root / "masks.bin", "rb"),
                "metadata": open(self.data_root / "metadata.bin", "rb"),
            }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        读取第 idx 个样本
        
        返回格式与 train_distill_single_test.py 完全兼容：
        - image: (3, 1024, 1024) uint8（未归一化，训练脚本会处理）
        - teacher_features: (256, 64, 64) float32（P4_S16，已 squeeze(0)）
        - edge_256x256: (256, 256) float32
        - fg_map: (1, 64, 64) float32（对应 P4_S16 的尺寸）
        - bg_map: (1, 64, 64) float32
        - edge_map: (1, 64, 64) float32（从 edge_256x256 下采样）
        - box_prompts_xyxy: (N, 4) float32 或 empty(0, 4)
        - box_prompts_masks_orig: (N, 256, 256) float32 或 empty(0, 256, 256)
        - box_prompts_count: int
        - box_prompts_flag: int (0 或 1)
        - box_prompts_meta: list
        - geometry_color_flag: int
        - has_bbox: bool
        - num_bboxes: int
        - image_shape: np.ndarray (3,) int32 [H, W, C]
        - image_id: str（如果 keys.txt 存在）
        """
        self._init_files()
        
        # === 1. 读取图像 (Image) ===
        self.files["images"].seek(idx * self.SIZE_IMG)
        img_bytes = self.files["images"].read(self.SIZE_IMG)
        if len(img_bytes) != self.SIZE_IMG:
            raise IOError(f"Failed to read image at index {idx}: expected {self.SIZE_IMG} bytes, got {len(img_bytes)}")
        
        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
            self.IMG_SIZE, self.IMG_SIZE, 3
        )
        # 转换为 CHW 格式，保持 uint8（训练脚本会处理归一化）
        img_tensor = torch.from_numpy(img_np.copy()).permute(2, 0, 1)  # (3, 1024, 1024)
        
        # === 2. 读取特征 (Teacher Features: P4_S16 + P5_S32) ===
        self.files["features"].seek(idx * self.SIZE_FEAT)
        feat_bytes = self.files["features"].read(self.SIZE_FEAT)
        if len(feat_bytes) != self.SIZE_FEAT:
            raise IOError(f"Failed to read features at index {idx}")
        
        # 切分 P4 和 P5
        p4_bytes = feat_bytes[:self.SIZE_P4]
        p5_bytes = feat_bytes[self.SIZE_P4:]
        
        p4_np = np.frombuffer(p4_bytes, dtype=np.float32).reshape(1, 256, 64, 64)
        p5_np = np.frombuffer(p5_bytes, dtype=np.float32).reshape(1, 256, 32, 32)
        
        # 转换为 Tensor，并 squeeze(0) 以匹配训练脚本期望的格式
        # 训练脚本期望: teacher_features = (256, 64, 64)，不是 (1, 256, 64, 64)
        teacher_features = torch.from_numpy(p4_np.copy()).squeeze(0)  # (256, 64, 64)
        
        # === 3. 读取边缘图 (Edge Maps) ===
        self.files["edge_maps"].seek(idx * self.SIZE_EDGE_ALL)
        edge_bytes = self.files["edge_maps"].read(self.SIZE_EDGE_ALL)
        if len(edge_bytes) != self.SIZE_EDGE_ALL:
            raise IOError(f"Failed to read edge maps at index {idx}")
        
        # 切分 edge_256x256, edge_64x64, edge_32x32
        edge_256_bytes = edge_bytes[:self.SIZE_EDGE_256]
        edge_64_bytes = edge_bytes[self.SIZE_EDGE_256:self.SIZE_EDGE_256 + self.SIZE_EDGE_64]
        edge_32_bytes = edge_bytes[self.SIZE_EDGE_256 + self.SIZE_EDGE_64:]
        
        edge_256_np = np.frombuffer(edge_256_bytes, dtype=np.uint8).reshape(256, 256)
        edge_64_np = np.frombuffer(edge_64_bytes, dtype=np.uint8).reshape(64, 64)
        edge_32_np = np.frombuffer(edge_32_bytes, dtype=np.uint8).reshape(32, 32)
        
        # 转换为 float32（训练脚本期望 float32，格式为 (256, 256)，不是 (1, 256, 256)）
        # 与 _parse_npz_data_fast 保持一致：edge_256x256 = torch.from_numpy(edge_np).float().contiguous()
        edge_256x256 = torch.from_numpy(edge_256_np.copy()).float().contiguous()  # (256, 256)
        
        # === 4. 读取权重图 (Weight Maps) ===
        self.files["weight_maps"].seek(idx * self.SIZE_WEIGHT_ALL)
        weight_bytes = self.files["weight_maps"].read(self.SIZE_WEIGHT_ALL)
        if len(weight_bytes) != self.SIZE_WEIGHT_ALL:
            raise IOError(f"Failed to read weight maps at index {idx}")
        
        # 切分: fg_128, bg_128, fg_64, bg_64, fg_32, bg_32
        offset = 0
        fg_128 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_128], dtype=np.float32).reshape(128, 128)
        offset += self.SIZE_WEIGHT_128
        bg_128 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_128], dtype=np.float32).reshape(128, 128)
        offset += self.SIZE_WEIGHT_128
        fg_64 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_64], dtype=np.float32).reshape(64, 64)
        offset += self.SIZE_WEIGHT_64
        bg_64 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_64], dtype=np.float32).reshape(64, 64)
        offset += self.SIZE_WEIGHT_64
        fg_32 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_32], dtype=np.float32).reshape(32, 32)
        offset += self.SIZE_WEIGHT_32
        bg_32 = np.frombuffer(weight_bytes[offset:offset+self.SIZE_WEIGHT_32], dtype=np.float32).reshape(32, 32)
        
        # 训练脚本期望 fg_map 和 bg_map 对应 P4_S16 的尺寸 (64x64)
        fg_map = torch.from_numpy(fg_64.copy()).unsqueeze(0)  # (1, 64, 64)
        bg_map = torch.from_numpy(bg_64.copy()).unsqueeze(0)  # (1, 64, 64)
        
        # edge_map 从 edge_256x256 下采样到 64x64（对应 P4_S16）
        # 使用 torch 的 interpolate（训练脚本中也有类似逻辑）
        import torch.nn.functional as F
        edge_map = edge_256x256.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
        edge_map = F.interpolate(edge_map, size=(64, 64), mode='area')  # (1, 1, 64, 64)
        edge_map = edge_map.squeeze(0)  # (1, 64, 64)
        
        # === 5. 读取掩码 (Masks) ===
        self.files["masks"].seek(idx * self.SIZE_MASK)
        mask_bytes = self.files["masks"].read(self.SIZE_MASK)
        if len(mask_bytes) != self.SIZE_MASK:
            raise IOError(f"Failed to read mask at index {idx}")
        
        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(1, self.MASK_SIZE, self.MASK_SIZE)
        # 转换为 float32（训练脚本期望 float32）
        mask = torch.from_numpy(mask_np.copy()).float()  # (1, 256, 256)
        
        # === 6. 读取边界框 (BBoxes) ===
        self.files["bboxes"].seek(idx * self.SIZE_BBOX)
        bbox_bytes = self.files["bboxes"].read(self.SIZE_BBOX)
        if len(bbox_bytes) != self.SIZE_BBOX:
            raise IOError(f"Failed to read bbox at index {idx}")
        
        bbox_np = np.frombuffer(bbox_bytes, dtype=np.float32).reshape(1, 4)
        bbox = torch.from_numpy(bbox_np.copy())  # (1, 4)
        
        # === 7. 读取元数据 (Metadata) ===
        self.files["metadata"].seek(idx * self.SIZE_META)
        meta_bytes = self.files["metadata"].read(self.SIZE_META)
        if len(meta_bytes) != self.SIZE_META:
            raise IOError(f"Failed to read metadata at index {idx}")
        
        # [num_bboxes, has_bbox, H, W, C]
        meta = np.frombuffer(meta_bytes, dtype=np.int32)
        num_bboxes = int(meta[0])
        has_bbox = bool(meta[1])
        image_shape = meta[2:5].copy()  # [H, W, C]
        
        # === 8. 构造 box_prompts（与训练脚本兼容） ===
        # 训练脚本期望的格式：
        # - box_prompts_xyxy: (N, 4) float32 或 empty(0, 4)
        # - box_prompts_masks_orig: (N, 256, 256) float32 或 empty(0, 256, 256)
        # - box_prompts_count: int
        # - box_prompts_flag: int (0 或 1)
        # - box_prompts_meta: list
        
        if num_bboxes > 0 and has_bbox:
            # 有框：使用读取的 bbox 和 mask
            # 注意：bbox 是 XYWH 格式，需要转换为 XYXY（训练脚本期望 XYXY）
            # 但转换脚本已经保存了 XYXY 格式（从 NPZ 读取时已转换）
            # 所以这里直接使用
            box_prompts_xyxy = bbox  # (1, 4) 已经是 XYXY 格式
            box_prompts_masks_orig = mask  # (1, 256, 256)
            box_prompts_count = 1
            box_prompts_flag = 1
        else:
            # 无框：返回空 tensor
            box_prompts_xyxy = torch.empty(0, 4, dtype=torch.float32)
            box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
            box_prompts_count = 0
            box_prompts_flag = 0
        
        box_prompts_meta = []  # 通常为空
        
        # === 9. 其他字段 ===
        geometry_color_flag = 0  # 默认值（转换脚本未保存此字段）
        
        # === 10. 获取 image_id（如果存在） ===
        image_id = None
        if self.image_ids is not None and idx < len(self.image_ids):
            image_id = self.image_ids[idx]
        else:
            # 如果没有 keys.txt，使用索引生成 ID
            image_id = f"sample_{idx:06d}"
        
        # === 构造返回字典（与 train_distill_single_test.py 完全兼容） ===
        return {
            "image": img_tensor,  # (3, 1024, 1024) uint8
            "teacher_features": teacher_features,  # (256, 64, 64) float32
            "edge_256x256": edge_256x256,  # (256, 256) float32
            "fg_map": fg_map,  # (1, 64, 64) float32
            "bg_map": bg_map,  # (1, 64, 64) float32
            "edge_map": edge_map,  # (1, 64, 64) float32
            "box_prompts_xyxy": box_prompts_xyxy,  # (N, 4) 或 (0, 4)
            "box_prompts_masks_orig": box_prompts_masks_orig,  # (N, 256, 256) 或 (0, 256, 256)
            "box_prompts_count": box_prompts_count,
            "box_prompts_flag": box_prompts_flag,
            "box_prompts_meta": box_prompts_meta,
            "geometry_color_flag": geometry_color_flag,
            "has_bbox": has_bbox,
            "num_bboxes": num_bboxes,
            "image_shape": image_shape,  # np.ndarray (3,) int32
            "image_id": image_id,
        }
    
    def __del__(self):
        """清理时关闭所有文件句柄"""
        for f in self.files.values():
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self.files.clear()


#!/usr/bin/env python3
"""
BinaryDistillDataset V2 (Hybrid Storage + Dynamic Protocol)

特性摘要：
1. 图像：images.idx.npy (offset,length) + images.bin 变长存储，无填充浪费。
2. 特征：config.json 驱动的定长布局，支持任意 stride / 通道 / 输入尺寸。
3. Edge & Weight：按需写入，一键扩展，完全消除硬编码常量。
"""

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryDistillDatasetV2(Dataset):
    """面向多 Teacher、多尺度蒸馏场景的混合存储读取器。"""

    SIZE_BBOX = 4 * 4          # float32 xyxy
    SIZE_MASK = 256 * 256      # 单通道 uint8
    SIZE_META = 5 * 4          # int32: [num, has, H, W, C]

    def __init__(self, data_root: str, verbose: bool = True) -> None:
        self.data_root = Path(data_root)
        self.verbose = verbose
        self.files: Dict[str, Optional[object]] = {}

        self._load_config()
        self._load_image_index()
        self._calculate_fixed_sizes()
        self._verify_files()
        self._load_keys()

        if self.verbose:
            print(f"✅ [BinaryDataset V2] Root: {self.data_root}")
            print(f"   - Samples: {self.num_samples}")
            print(f"   - Resolution: {self.input_h}x{self.input_w}")
            print(f"   - Feature Dim: {self.feat_dim}")
            print(f"   - Strides: {self.strides}")
            print(f"   - Edge Maps: {self.save_edge}, Weight Maps: {self.save_weight}")

    # ------------------------------------------------------------------ #
    # 配置与索引
    # ------------------------------------------------------------------ #
    def _load_config(self) -> None:
        cfg_path = self.data_root / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as fp:
            self.cfg = json.load(fp)

        self.num_samples = self.cfg["dataset_info"]["total_samples"]
        res_cfg = self.cfg["resolution"]["input_size"]
        if isinstance(res_cfg, (list, tuple)):
            self.input_h, self.input_w = res_cfg
        else:
            self.input_h = self.input_w = int(res_cfg)

        self.feat_dim = self.cfg["storage"]["feature_dim"]
        self.strides = sorted(self.cfg["storage"]["strides"])
        feat_dtype = self.cfg["storage"].get("feature_dtype", "float32")
        if feat_dtype != "float32":
            raise ValueError(f"仅支持 float32 特征，收到 {feat_dtype}")
        self.save_edge = self.cfg["storage"].get("save_edge", True)
        self.save_weight = self.cfg["storage"].get("save_weight", True)

    def _load_image_index(self) -> None:
        idx_path = self.data_root / "images.idx.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"Image index missing: {idx_path}")

        self.img_indices = np.load(idx_path, mmap_mode="r")
        if self.img_indices.shape[1] != 2:
            raise ValueError("images.idx.npy 应为 (N, 2) [offset, length]")
        if self.img_indices.shape[0] != self.num_samples:
            print(f"[Warning] Index count {self.img_indices.shape[0]} != Config {self.num_samples}")

    def _load_keys(self) -> None:
        keys_path = self.data_root / "keys.txt"
        self.image_ids = None
        if keys_path.exists():
            with open(keys_path, "r", encoding="utf-8") as fp:
                self.image_ids = [line.strip() for line in fp if line.strip()]

    # ------------------------------------------------------------------ #
    # 偏移计算
    # ------------------------------------------------------------------ #
    def _calculate_fixed_sizes(self) -> None:
        self.feat_shapes: Dict[int, tuple] = {}
        self.feat_offsets: Dict[int, int] = {}
        self.feat_bytes: Dict[int, int] = {}

        total = 0
        for stride in self.strides:
            h, w = self.input_h // stride, self.input_w // stride
            size = self.feat_dim * h * w * 4
            self.feat_shapes[stride] = (self.feat_dim, h, w)
            self.feat_offsets[stride] = total
            self.feat_bytes[stride] = size
            total += size
        self.bytes_per_sample_feat = total

        self.edge_shapes: Dict[int, tuple] = {}
        self.edge_offsets: Dict[int, int] = {}
        self.edge_bytes: Dict[int, int] = {}

        total = 0
        if self.save_edge:
            for stride in self.strides:
                h, w = self.input_h // stride, self.input_w // stride
                size = h * w  # uint8
                self.edge_shapes[stride] = (h, w)
                self.edge_offsets[stride] = total
                self.edge_bytes[stride] = size
                total += size
        self.bytes_per_sample_edge = total

        self.weight_shapes: Dict[int, tuple] = {}
        self.weight_offsets: Dict[int, int] = {}
        self.weight_bytes: Dict[int, int] = {}

        total = 0
        if self.save_weight:
            for stride in self.strides:
                h, w = self.input_h // stride, self.input_w // stride
                size = h * w * 2 * 4  # fg + bg, float32
                self.weight_shapes[stride] = (h, w)
                self.weight_offsets[stride] = total
                self.weight_bytes[stride] = size
                total += size
        self.bytes_per_sample_weight = total

    # ------------------------------------------------------------------ #
    # 文件校验与初始化
    # ------------------------------------------------------------------ #
    def _verify_files(self) -> None:
        required = ["images.bin", "features.bin", "bboxes.bin", "masks.bin", "metadata.bin"]
        if self.save_edge:
            required.append("edge_maps.bin")
        if self.save_weight:
            required.append("weight_maps.bin")

        for fname in required:
            if not (self.data_root / fname).exists():
                raise FileNotFoundError(f"Missing required file: {fname}")

    def _init_files(self) -> None:
        if self.files:
            return
        self.files = {
            "images": open(self.data_root / "images.bin", "rb"),
            "features": open(self.data_root / "features.bin", "rb"),
            "bboxes": open(self.data_root / "bboxes.bin", "rb"),
            "masks": open(self.data_root / "masks.bin", "rb"),
            "metadata": open(self.data_root / "metadata.bin", "rb"),
        }
        if self.save_edge:
            self.files["edge_maps"] = open(self.data_root / "edge_maps.bin", "rb")
        if self.save_weight:
            self.files["weight_maps"] = open(self.data_root / "weight_maps.bin", "rb")

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_files()

        image = self._read_image(idx)
        features = self._read_features(idx)
        edge_maps = self._read_edge_maps(idx) if self.save_edge else {}
        weight_maps = self._read_weight_maps(idx) if self.save_weight else {}
        bbox, mask, meta = self._read_boxes_masks_meta(idx)

        sample = {
            "image": image,
            "features": features,
            "edge_maps": edge_maps,
            "weight_maps": weight_maps,
            "box_prompts_xyxy": bbox,
            "box_prompts_masks_orig": mask.float(),
            "has_bbox": bool(meta[1]),
            "image_shape": torch.from_numpy(meta[2:5].copy()),
            "image_id": self.image_ids[idx] if self.image_ids else str(idx),
        }
        return sample

    # ------------------------------------------------------------------ #
    # 具体读取逻辑
    # ------------------------------------------------------------------ #
    def _read_image(self, idx: int) -> torch.Tensor:
        offset, length = self.img_indices[idx]
        handle = self.files["images"]
        handle.seek(int(offset))
        jpg_bytes = handle.read(int(length))

        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def _read_features(self, idx: int) -> Dict[str, torch.Tensor]:
        handle = self.files["features"]
        handle.seek(idx * self.bytes_per_sample_feat)
        blob = handle.read(self.bytes_per_sample_feat)
        if len(blob) != self.bytes_per_sample_feat:
            raise IOError(f"Feature blob truncated @ idx={idx}")

        feats = {}
        for stride in self.strides:
            start = self.feat_offsets[stride]
            end = start + self.feat_bytes[stride]
            chunk = blob[start:end]
            c, h, w = self.feat_shapes[stride]
            arr = np.frombuffer(chunk, dtype=np.float32).reshape(c, h, w)
            feats[f"S{stride}"] = torch.from_numpy(arr.copy())
        return feats

    def _read_edge_maps(self, idx: int) -> Dict[str, torch.Tensor]:
        handle = self.files["edge_maps"]
        handle.seek(idx * self.bytes_per_sample_edge)
        blob = handle.read(self.bytes_per_sample_edge)
        if len(blob) != self.bytes_per_sample_edge:
            raise IOError(f"Edge blob truncated @ idx={idx}")

        edges = {}
        for stride in self.strides:
            start = self.edge_offsets.get(stride)
            if start is None:
                continue
            end = start + self.edge_bytes[stride]
            chunk = blob[start:end]
            h, w = self.edge_shapes[stride]
            arr = np.frombuffer(chunk, dtype=np.uint8).reshape(h, w)
            edges[f"S{stride}"] = torch.from_numpy(arr.copy()).float()
        return edges

    def _read_weight_maps(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        handle = self.files["weight_maps"]
        handle.seek(idx * self.bytes_per_sample_weight)
        blob = handle.read(self.bytes_per_sample_weight)
        if len(blob) != self.bytes_per_sample_weight:
            raise IOError(f"Weight blob truncated @ idx={idx}")

        weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for stride in self.strides:
            start = self.weight_offsets.get(stride)
            if start is None:
                continue
            end = start + self.weight_bytes[stride]
            chunk = blob[start:end]
            data = np.frombuffer(chunk, dtype=np.float32)
            h, w = self.weight_shapes[stride]
            mid = data.size // 2
            fg = data[:mid].reshape(1, h, w)
            bg = data[mid:].reshape(1, h, w)
            weights[f"S{stride}"] = {
                "fg": torch.from_numpy(fg.copy()),
                "bg": torch.from_numpy(bg.copy()),
            }
        return weights

    def _read_boxes_masks_meta(self, idx: int):
        bbox_file = self.files["bboxes"]
        mask_file = self.files["masks"]
        meta_file = self.files["metadata"]

        bbox_file.seek(idx * self.SIZE_BBOX)
        bbox = torch.from_numpy(
            np.frombuffer(bbox_file.read(self.SIZE_BBOX), dtype=np.float32).reshape(1, 4)
        )

        mask_file.seek(idx * self.SIZE_MASK)
        mask = torch.from_numpy(
            np.frombuffer(mask_file.read(self.SIZE_MASK), dtype=np.uint8).reshape(1, 256, 256)
        )

        meta_file.seek(idx * self.SIZE_META)
        meta = np.frombuffer(meta_file.read(self.SIZE_META), dtype=np.int32)
        return bbox, mask, meta

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #
    def __del__(self):
        for handle in self.files.values():
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
        self.files.clear()


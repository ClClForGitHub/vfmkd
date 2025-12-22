#!/usr/bin/env python3
"""
单独测试 MSE/FGD/FSDlike 蒸馏损失的最小训练脚本。
参考 VFMKD/tools/warmup_training_v1.py 的数据/日志风格，仅替换特征对齐损失。
"""

import os
import sys
import warnings

# 使用 conda 环境自带的 libstdc++，解决 GLIBCXX 版本问题
# 必须在导入任何可能依赖 libstdc++ 的模块（如 PIL、torch 等）之前设置
if 'CONDA_PREFIX' in os.environ:
    try:
        conda_prefix = os.environ['CONDA_PREFIX']
        conda_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(conda_lib):
            # 将 conda 的 lib 目录添加到 LD_LIBRARY_PATH 的最前面（最高优先级）
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if conda_lib not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib
            
            # 显式加载 conda 环境的 libstdc++，确保后续导入的模块使用正确的版本
            try:
                import ctypes
                libstdcxx_path = os.path.join(conda_lib, 'libstdc++.so.6')
                if os.path.exists(libstdcxx_path):
                    # 使用 RTLD_GLOBAL 标志，让所有后续加载的库都能使用这个 libstdc++
                    ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                # 如果加载失败，至少确保 LD_LIBRARY_PATH 已设置
                pass
    except Exception as e:
        # 捕获所有异常，防止脚本在系统库加载阶段就崩溃
        print(f"Warning: Failed to preload conda libstdc++: {e}")
import argparse
import time
import logging
import contextlib
from pathlib import Path
from typing import List
from datetime import datetime
from zipfile import BadZipFile
import tarfile
import json
import hashlib
import multiprocessing
import shutil
import tempfile
import threading
import queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

# 屏蔽Windows/Numpy等运行时告警，保持训练输出干净
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# 【内存优化】在导入torch之前设置CUDA内存分配策略，避免训练后期OOM
# 设置max_split_size_mb来减少内存碎片（必须在导入torch之前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# 添加项目根目录到路径（从tools/core/exper/向上三级到VFMKD/）
_script_dir = os.path.dirname(os.path.abspath(__file__))  # tools/core/exper/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))  # VFMKD/
sys.path.insert(0, project_root)

# 添加 RT-DETR v2 源码目录，供后续直接 import
_rtdetr_src_dir = os.path.join(project_root, "tools", "core", "RT-DETR-main", "RT-DETR-main", "rtdetrv2_pytorch")
if os.path.isdir(_rtdetr_src_dir) and _rtdetr_src_dir not in sys.path:
    sys.path.insert(0, _rtdetr_src_dir)

# 导入二进制数据集类
from tools.core.exper.binary_dataset import BinaryDistillDataset

# 延迟导入：只在真正需要时导入模块，避免在显示帮助信息时就触发导入错误
# 这些导入会在 DistillSingleTester 类初始化时进行
import importlib

# 定义导入函数，延迟到真正需要时再导入
class RTDETRBackboneAdapter(nn.Module):
    """统一 RT-DETR v2 backbone 的输出格式与通道信息接口。"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        elif isinstance(feats, torch.Tensor):
            feats = [feats]
        elif isinstance(feats, tuple):
            feats = list(feats)
        elif isinstance(feats, list):
            pass
        else:
            raise TypeError(f"Unexpected feature type from backbone: {type(feats)}")
        return tuple(feats)

    def get_feature_channels(self) -> dict[str, int] | None:
        channel_attrs = ("out_channels", "channels", "_out_channels")
        channels = None
        for attr in channel_attrs:
            if hasattr(self.backbone, attr):
                channels = getattr(self.backbone, attr)
                break
        if channels is None:
            return None
        channels = list(channels)
        level_keys = ["s4", "s8", "s16", "s32"]
        return {
            level_keys[idx]: int(ch)
            for idx, ch in enumerate(channels)
            if idx < len(level_keys)
        }


def _import_backbones():
    """延迟导入 RT-DETR v2 backbone 构建函数。"""
    try:
        from src.nn.backbone import PResNet, HGNetv2, CSPResNet, TimmModel
    except ImportError as e:
        raise ImportError(f"导入 RT-DETR v2 backbone 模块失败: {e}") from e

    default_return_idx = [0, 1, 2, 3]

    def _normalize_pretrained_path(path: str | None) -> str | None:
        if not path:
            return None
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            return path
        return str(Path(path).expanduser())

    def build_rtdetrv2_backbone(config: dict) -> nn.Module:
        backbone_type = config.get("student_backbone_type", "PResNet")
        return_idx = config.get("student_return_idx") or default_return_idx
        return_idx = [int(idx) for idx in return_idx]
        pretrained_path = _normalize_pretrained_path(config.get("student_pretrained"))
        freeze_norm = bool(config.get("student_backbone_freeze_norm", False))
        depth = int(config.get("student_backbone_depth", 50))
        variant = config.get("student_backbone_variant", "L")

        if backbone_type == "PResNet":
            presnet_variant = config.get("student_presnet_variant", "d")
            model = PResNet(
                depth=depth,
                variant=presnet_variant,
                num_stages=4,
                return_idx=return_idx,
                act="relu",
                freeze_norm=freeze_norm,
                pretrained=pretrained_path or False,
            )
        elif backbone_type == "HGNetv2":
            model = HGNetv2(
                name=variant.upper(),
                return_idx=return_idx,
                freeze_norm=freeze_norm,
                pretrained=pretrained_path or False,
            )
        elif backbone_type == "CSPResNet":
            model = CSPResNet(
                name=variant.lower(),
                return_idx=return_idx,
                pretrained=pretrained_path or False,
            )
        elif backbone_type == "TimmModel":
            return_layers = config.get(
                "student_timm_return_layers",
                ["layer1", "layer2", "layer3", "layer4"],
            )
            timm_name = variant if variant != "L" else "resnet50"
            model = TimmModel(
                name=timm_name,
                return_layers=return_layers,
                pretrained=True if pretrained_path else False,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        return RTDETRBackboneAdapter(model)

    return build_rtdetrv2_backbone

def _import_other_modules():
    """延迟导入其他模块"""
    edge_head_module = importlib.import_module('vfmkd.models.heads.edge_head')
    UniversalEdgeHead = edge_head_module.UniversalEdgeHead
    feature_loss_module = importlib.import_module('vfmkd.distillation.losses.feature_loss')
    FeatureLoss = feature_loss_module.FeatureLoss
    fgd_loss_module = importlib.import_module('vfmkd.distillation.losses.fgd_loss')
    FGDLoss = fgd_loss_module.FGDLoss
    fsd_loss_module = importlib.import_module('vfmkd.distillation.losses.fsd_loss')
    FSDLikeLoss = fsd_loss_module.FSDLikeLoss
    edge_loss_module = importlib.import_module('vfmkd.distillation.losses.edge_loss')
    EdgeDistillationLoss = edge_loss_module.EdgeDistillationLoss
    adapters_module = importlib.import_module('vfmkd.distillation.adapters')
    SimpleAdapter = adapters_module.SimpleAdapter
    EdgeAdapter = adapters_module.EdgeAdapter
    SimpleAdapterStatic = getattr(adapters_module, 'SimpleAdapterStatic', None)
    EdgeAdapterStatic = getattr(adapters_module, 'EdgeAdapterStatic', None)
    gt_adapter_module = importlib.import_module('vfmkd.distillation.gt_adapter')
    build_batch_bboxes_from_ids = gt_adapter_module.build_batch_bboxes_from_ids
    build_fg_bg_from_ids = gt_adapter_module.build_fg_bg_from_ids
    prompt_heads_module = importlib.import_module('tools.core.prompt.heads')
    load_sam2_heads = prompt_heads_module.load_sam2_heads
    return {
        'UniversalEdgeHead': UniversalEdgeHead,
        'FeatureLoss': FeatureLoss,
        'FGDLoss': FGDLoss,
        'FSDLikeLoss': FSDLikeLoss,
        'EdgeDistillationLoss': EdgeDistillationLoss,
        'SimpleAdapter': SimpleAdapter,
        'EdgeAdapter': EdgeAdapter,
        'SimpleAdapterStatic': SimpleAdapterStatic,
        'EdgeAdapterStatic': EdgeAdapterStatic,
        'build_batch_bboxes_from_ids': build_batch_bboxes_from_ids,
        'build_fg_bg_from_ids': build_fg_bg_from_ids,
        'load_sam2_heads': load_sam2_heads,
    }

# 全局变量，在首次使用时初始化
_RTDETRBackboneBuilder = None
_OTHER_MODULES = None

logger = logging.getLogger("distill_single_test")

# torchmetrics 用于优化的评估指标计算（在 logger 定义后导入）
try:
    from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, CosineSimilarity
    from torchmetrics.classification import BinaryJaccardIndex
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    logger.warning("torchmetrics 未安装，evaluate 将使用手动指标计算（性能较低）")

# ==================== 环境配置 ====================
@torch.no_grad()
def infer_feature_channels(backbone: nn.Module, device: torch.device, img_size: int = 1024) -> tuple[int, int]:
    """推断S4与S16的通道数；优先调用backbone提供的接口，否则用dummy前向一次。"""
    if hasattr(backbone, 'get_feature_channels'):
        ch = backbone.get_feature_channels()
        # 约定键名
        s4_c = ch.get('s4') or ch.get('P2') or list(ch.values())[0]
        s16_c = ch.get('s16') or ch.get('P4') or list(ch.values())[2]
        return int(s4_c), int(s16_c)
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with amp.autocast(enabled=(device.type == 'cuda')):
        feats = backbone(x)
    return int(feats[0].shape[1]), int(feats[2].shape[1])


def get_env_config(env_type: str = "ssh"):
    """
    根据环境类型返回数据集和输出路径配置
    
    Args:
        env_type: "ssh" 或 "local"
    
    Returns:
        dict: 包含features_dir, images_dir, test_features_dir, test_images_dir, output_base_dir的配置
    """
    if env_type == "ssh":
        return {
            "features_dir": "/home/team/zouzhiyuan/dataset/sa1b/extracted",  # npz_dir 模式使用
            "images_dir": "/home/team/zouzhiyuan/dataset/sa1b",
            "tar_shard_dir": "/home/team/zouzhiyuan/dataset/sa1b_tar_shards",  # tar_shard 模式使用
            "train_1200_features_dir": "/home/team/zouzhiyuan/dataset/sa1b/train_1200/extracted",
            "train_1200_images_dir": "/home/team/zouzhiyuan/dataset/sa1b/train_1200",
            "test_features_dir": "/home/team/zouzhiyuan/dataset/sa1b/test/extracted",
            "test_images_dir": "/home/team/zouzhiyuan/dataset/sa1b/test_resized_1024",
            "output_base_dir": "/home/team/zouzhiyuan/vfmkd/outputs",
        }
    elif env_type == "local":
        # TODO: 实现local环境的路径配置
        return {
            "features_dir": None,  # 需要用户指定
            "images_dir": None,   # 需要用户指定
            "test_features_dir": None,  # 需要用户指定
            "test_images_dir": None,   # 需要用户指定
            "output_base_dir": "outputs",
        }
    else:
        raise ValueError(f"Unsupported env_type: {env_type}, must be 'ssh' or 'local'")


def analyze_checkpoint_structure(checkpoint_path: str | Path):
    """
    分析保存的checkpoint结构，打印包含哪些权重
    
    Args:
        checkpoint_path: checkpoint文件路径
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    logger.info("%s", "")
    logger.info("%s", "=" * 60)
    logger.info("Checkpoint结构分析: %s", checkpoint_path)
    logger.info("%s", "=" * 60)
    
    # 基本信息
    if 'epoch' in checkpoint:
        logger.info("训练epoch: %s", checkpoint['epoch'])
    if 'config' in checkpoint:
        logger.info("配置信息: 已包含")
    
    # 权重模块
    logger.info("")
    logger.info("保存的权重模块:")
    for key in ['backbone', 'edge_adapter', 'edge_head', 'feature_adapter', 'optimizer']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            num_params = sum(p.numel() for p in state_dict.values()) if isinstance(state_dict, dict) else 0
            num_keys = len(state_dict) if isinstance(state_dict, dict) else 0
            logger.info("  ✓ %s: %s 个参数组, 总参数量: %s", key, num_keys, f"{num_params:,}")
            
            # 如果是适配器，列出内部的适配器
            if key in ['feature_adapter', 'edge_adapter']:
                adapter_keys = [k for k in state_dict.keys() if k.startswith('adapters.')]
                if adapter_keys:
                    logger.info("    内部适配器: %s 个", len(adapter_keys))
                    for adapter_key in adapter_keys[:5]:  # 只显示前5个
                        logger.info("      - %s", adapter_key)
                    if len(adapter_keys) > 5:
                        logger.info("      ... 还有 %s 个适配器", len(adapter_keys) - 5)
        else:
            logger.info("  ✗ %s: 未找到", key)
    
    logger.info("%s", "=" * 60)
    logger.info("")
    return checkpoint


def create_output_directory(loss_type: str, enable_edge_boost: bool, output_base_dir: Path, 
                           backbone: str = "rtdetrv2", extra_suffix: str = "") -> Path:
    """
    创建输出目录结构：
    {output_base_dir}/distill_single_test_{LOSS_TYPE}/{timestamp}_{backbone}_{edge_suffix}/
        - logs/
        - models/
        - visualizations/
    
    Args:
        loss_type: "mse", "fgd", "fsdlike"
        enable_edge_boost: 是否启用边缘增强
        output_base_dir: 输出根目录
        backbone: 骨干网络类型
        extra_suffix: 额外的后缀
    
    Returns:
        Path: 创建的输出目录路径
    """
    # 构建目录名
    loss_name = loss_type.upper()
    edge_suffix = "_edge_boost" if enable_edge_boost else "_no_edge_boost"
    if extra_suffix:
        edge_suffix += f"_{extra_suffix}"
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建目录路径
    base_name = f"distill_single_test_{loss_name}"
    subdir_name = f"{timestamp}_{backbone}{edge_suffix}"
    
    output_dir = output_base_dir / base_name / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "models" / "logs").mkdir(exist_ok=True)
    
    return output_dir


class StageTimer:
    def __init__(self, device: torch.device, stages: List[str]):
        self.is_cuda = device.type == "cuda"
        self.device = device
        self.records: dict[str, list] = {stage: [] for stage in stages}

    def start(self, stage: str):
        if stage not in self.records:
            self.records[stage] = []
        if self.is_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            self.records[stage].append((start_evt, end_evt))
            return end_evt
        start_time = time.time()
        self.records[stage].append([start_time, None])
        return None

    def stop(self, stage: str, handle=None) -> None:
        if stage not in self.records or not self.records[stage]:
            return
        if self.is_cuda:
            if handle is not None:
                handle.record()
        else:
            self.records[stage][-1][1] = time.time()

    def durations(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        if self.is_cuda:
            has_records = any(self.records[stage] for stage in self.records)
            if has_records:
                torch.cuda.synchronize()
            for stage, pairs in self.records.items():
                total_ms = 0.0
                for start_evt, end_evt in pairs:
                    # 检查 end_event 是否已经被记录
                    try:
                        # 如果 end_event 还没有被记录，query() 会返回 False
                        if end_evt.query():
                            total_ms += start_evt.elapsed_time(end_evt)
                        # 如果 end_event 还没记录，跳过这个计时（可能 stop 没有被调用）
                    except RuntimeError:
                        # 如果查询失败，说明 event 还没准备好，跳过
                        pass
                totals[stage] = total_ms / 1000.0
        else:
            for stage, pairs in self.records.items():
                total = 0.0
                for start, end in pairs:
                    end_time = end if end is not None else time.time()
                    total += end_time - start
                totals[stage] = total
        return totals

    def reset(self) -> None:
        for stage in self.records:
            self.records[stage].clear()


class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device=device) if self.is_cuda else None
        self.next_batch = None
        self._preload_count = 0
        self._preload()

    def _move_to_device(self, batch: dict) -> dict:
        out: dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _preload(self) -> None:
        try:
            batch = next(self.loader)
            self._preload_count += 1
            if self._preload_count == 1:
                logger.info("[DATA LOADER] 成功预加载第一个batch，数据加载器正常工作")
        except StopIteration:
            self.next_batch = None
            if self._preload_count == 0:
                logger.error("[DATA LOADER] ❌ 数据加载器在第一次预加载时就耗尽（StopIteration）")
                logger.error("[DATA LOADER] 可能原因:")
                logger.error("  1. 数据集为空或没有有效样本")
                logger.error("  2. tar文件读取失败或格式问题")
                logger.error("  3. 数据加载器配置问题（num_workers等）")
            return

        if not self.is_cuda:
            self.next_batch = batch
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self._move_to_device(batch)

    def next(self) -> dict | None:
        if self.next_batch is None:
            return None

        if self.is_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self.next_batch
        self._preload()
        return batch


def _loader_worker_init(_worker_id: int) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _custom_collate_fn(batch: list[dict]) -> dict:
    """
    自定义 collate 函数，处理变长张量（box_prompts 等）
    
    对于变长字段，保持为列表而不是 stack，避免形状不匹配错误
    注意：bboxes 和 masks 已移除（未在训练中使用）
    """
    from torch.utils.data._utils.collate import default_collate
    
    if not batch:
        return {}
    
    # 变长字段列表（需要特殊处理，保持为列表）
    variable_length_fields = {
        'box_prompts_xyxy',
        'box_prompts_masks_orig',
        'box_prompts_meta',  # 已经是列表，不需要 collate
    }
    
    # 获取所有键
    all_keys = list(batch[0].keys())
    result = {}
    
    # 逐个字段处理
    for key in all_keys:
        field_values = [sample[key] for sample in batch]
        
        if key in variable_length_fields:
            # 变长字段：保持为列表
            result[key] = field_values
        else:
            # 固定字段：使用默认 collate（对单个字段的列表）
            try:
                result[key] = default_collate(field_values)
            except Exception as e:
                # 如果默认 collate 失败（例如字符串列表），保持为列表
                logger.warning(f"字段 {key} 使用默认 collate 失败，保持为列表: {e}")
                result[key] = field_values
    
    return result


class CorruptNPZError(RuntimeError):
    def __init__(self, file_path: str, message: str | None = None):
        super().__init__(message or f"Corrupted NPZ file: {file_path}")
        self.file_path = file_path


def _parse_npz_data(npz_data, verbose: bool = False):
    """
    【兼容性函数】保留原有接口，内部调用极速版本
    """
    return _parse_npz_data_fast(npz_data)


def _parse_npz_data_fast(npz_data):
    """
    【商业级极速解析】
    
    去除所有 try-except 和 if check。
    假设数据生产端（ETL）保证数据完整性。任何缺失都会直接触发 KeyError 导致训练崩溃（Fail Fast）。
    
    注意：已移除 bboxes 和 masks 字段（未在训练中使用，减少 I/O 开销）
    """
    # 1. 核心特征 (直接读取，不做分支判断)
    # 假设 ETL 统一输出了 P4_S16
    teacher_np = npz_data["P4_S16"]
    teacher_features = torch.from_numpy(teacher_np).squeeze(0).contiguous()
    
    # 2. 边缘图 (必须存在)
    edge_np = npz_data["edge_256x256"]
    # 确保转换为 float32（与 TarShardNPZDataset 保持一致）
    if edge_np.dtype != np.float32:
        edge_np = edge_np.astype(np.float32, copy=False)
    edge_256x256 = torch.from_numpy(edge_np).float().contiguous()
    
    # 获取维度用于拼接 key
    Hf, Wf = teacher_features.shape[-2], teacher_features.shape[-1]
    
    # 3. 辅助 Map (直接构造 Key，失败即崩)
    fg_tensor = torch.from_numpy(npz_data[f"fg_map_{Hf}x{Wf}"]).unsqueeze(0)
    bg_tensor = torch.from_numpy(npz_data[f"bg_map_{Hf}x{Wf}"]).unsqueeze(0)
    
    # edge_map 可能是可选的（某些数据可能没有）
    # 如果不存在，从 edge_256x256 实时下采样生成（与 extract_features_v1.py 方法一致）
    edge_key = f"edge_map_{Hf}x{Wf}"
    if edge_key in npz_data:
        edge_tensor = torch.from_numpy(npz_data[edge_key]).unsqueeze(0)
    elif "edge_256x256" in npz_data and cv2 is not None:
        # 实时下采样生成（与 extract_features_v1.py 和 TarShardNPZDataset 保持一致）
        edge_256 = npz_data["edge_256x256"]  # (256, 256) uint8 {0, 1}
        # 转为 float32（与 extract_features_v1.py 一致）
        edge_float = edge_256.astype(np.float32)
        # 使用 cv2.resize + INTER_AREA 下采样（与 extract_features_v1.py 完全一致）
        edge_resized = cv2.resize(
            edge_float,
            (Wf, Hf),  # 注意：cv2.resize 是 (width, height)
            interpolation=cv2.INTER_AREA
        )
        # 阈值化：> 0 则为 1.0，保持 float32（训练时需要 float32）
        edge_binary = (edge_resized > 0).astype(np.float32)
        edge_tensor = torch.from_numpy(edge_binary).unsqueeze(0)
    else:
        # 如果都没有，创建一个空的tensor
        edge_tensor = torch.empty(0, dtype=torch.float32)
    
    # 4. Box Prompts (可选字段，某些样本可能没有)
    # 优先使用预处理好的 box_prompts_*，否则从 bboxes 和 masks 转换
    # 初始化 image_shape（可能在后续分支中定义）
    image_shape = None
    
    if "box_prompts_xyxy" in npz_data and "box_prompts_masks_orig" in npz_data:
        # 已有预处理好的 box_prompts
        box_prompts_xyxy = torch.from_numpy(npz_data["box_prompts_xyxy"])
        if box_prompts_xyxy.ndim == 1:
            box_prompts_xyxy = box_prompts_xyxy.view(1, 4)
        
        box_prompts_masks_orig = torch.from_numpy(npz_data["box_prompts_masks_orig"])
        if box_prompts_masks_orig.ndim == 2:
            box_prompts_masks_orig = box_prompts_masks_orig.view(1, *box_prompts_masks_orig.shape)
        
        # 优先使用 NPZ 中的 box_prompts_count，否则从 box_prompts_xyxy 的形状推断
        if "box_prompts_count" in npz_data:
            box_prompts_count = int(npz_data["box_prompts_count"])
        else:
            # 从 box_prompts_xyxy 的形状推断（如果为空tensor则为0）
            box_prompts_count = box_prompts_xyxy.shape[0] if box_prompts_xyxy.numel() > 0 else 0
        
        # 根据 box_prompts_count 判断是否有框
        box_prompts_flag = 1 if box_prompts_count > 0 else 0
        
        # Metadata (Object array)
        if "box_prompts_meta" in npz_data:
            box_prompts_meta = npz_data["box_prompts_meta"].tolist()
        else:
            box_prompts_meta = []
        
        # 如果 npz_data 中有 image_shape，也读取它
        if "image_shape" in npz_data:
            image_shape = npz_data["image_shape"]
    elif "bboxes" in npz_data and "masks" in npz_data and "image_shape" in npz_data:
        # 从原始 bboxes 和 masks 转换（使用原图尺寸的框，不缩放）
        # 注意：NPZ中的bboxes是XYWH格式 [x, y, w, h]，需要转换为XYXY格式 [x0, y0, x1, y1]
        bboxes_orig = npz_data["bboxes"]  # [N, 4] XYWH格式，原图坐标系
        masks_orig = npz_data["masks"]  # [N, H_orig, W_orig] 或 object array
        image_shape = npz_data["image_shape"]  # [H, W, C] 或 [H, W] 或 shape=(3,)的一维数组
        
        # 获取原图尺寸（用于mask缩放）
        # image_shape 可能是 shape=(3,) 的一维数组 [H, W, C]，或 shape=(2,) 的 [H, W]
        if image_shape.size >= 2:
            orig_h, orig_w = int(image_shape[0]), int(image_shape[1])
        else:
            orig_h, orig_w = 1024, 1024  # 默认值
        
        # 转换bboxes：从XYWH格式转换为XYXY格式（原图坐标系，不缩放）
        # SAM2的prompt_encoder接受原图尺寸的框，内部会根据input_image_size处理
        if bboxes_orig.size > 0:
            bboxes_np = np.array(bboxes_orig, dtype=np.float32)
            if bboxes_np.ndim == 1:
                bboxes_np = bboxes_np.reshape(1, 4)
            
            # 转换XYWH -> XYXY: [x, y, w, h] -> [x0, y0, x1, y1]
            # x0 = x, y0 = y, x1 = x + w, y1 = y + h
            bboxes_xyxy = np.zeros_like(bboxes_np)
            bboxes_xyxy[:, 0] = bboxes_np[:, 0]  # x0 = x
            bboxes_xyxy[:, 1] = bboxes_np[:, 1]  # y0 = y
            bboxes_xyxy[:, 2] = bboxes_np[:, 0] + bboxes_np[:, 2]  # x1 = x + w
            bboxes_xyxy[:, 3] = bboxes_np[:, 1] + bboxes_np[:, 3]  # y1 = y + h
            
            box_prompts_xyxy = torch.from_numpy(bboxes_xyxy).float()
            box_prompts_count = bboxes_xyxy.shape[0]
        else:
            box_prompts_xyxy = torch.empty(0, 4, dtype=torch.float32)
            box_prompts_count = 0
        
        # 转换 masks：从原图尺寸缩放到256x256
        if masks_orig.size > 0 and box_prompts_count > 0:
            # 取第一个mask（假设每个样本只有一个mask）
            mask_orig = masks_orig[0] if hasattr(masks_orig, '__len__') and len(masks_orig) > 0 else masks_orig
            if isinstance(mask_orig, np.ndarray):
                # 转换为float32并缩放到256x256
                if mask_orig.dtype != np.float32:
                    mask_orig = mask_orig.astype(np.float32)
                
                # 使用cv2.resize缩放（如果可用）
                if cv2 is not None:
                    mask_256 = cv2.resize(mask_orig, (256, 256), interpolation=cv2.INTER_AREA)
                    # 二值化：>0则为1.0
                    mask_256 = (mask_256 > 0).astype(np.float32)
                else:
                    # 回退到torch的插值（更可靠，无需额外依赖）
                    mask_tensor = torch.from_numpy(mask_orig).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    mask_256_tensor = F.interpolate(mask_tensor, size=(256, 256), mode='nearest', align_corners=None)
                    mask_256 = (mask_256_tensor.squeeze().numpy() > 0).astype(np.float32)
                
                box_prompts_masks_orig = torch.from_numpy(mask_256).float().unsqueeze(0)  # [1, 256, 256]
            else:
                box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
                box_prompts_count = 0
        else:
            box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
            box_prompts_count = 0
        
        box_prompts_flag = 1 if box_prompts_count > 0 else 0
        box_prompts_meta = []
    else:
        # 既没有预处理好的 box_prompts，也没有原始的 bboxes/masks
        box_prompts_xyxy = torch.empty(0, 4, dtype=torch.float32)
        box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
        box_prompts_count = 0
        box_prompts_flag = 0
        box_prompts_meta = []
        # 如果 npz_data 中有 image_shape，也读取它
        if "image_shape" in npz_data:
            image_shape = npz_data["image_shape"]
    
    # 5. 其他标量（使用get提供默认值）
    geometry_color_flag = int(npz_data.get("geometry_color_flag", 0))
    has_bbox = bool(npz_data.get("has_bbox", False))
    num_bboxes = int(npz_data.get("num_bboxes", box_prompts_count))
    # box_prompts_flag 已在上面根据 box_prompts_count 正确设置，这里优先使用 NPZ 中的值（如果存在）
    # 但如果 NPZ 中的值与实际 count 不一致，以 count 为准
    if "box_prompts_flag" in npz_data:
        npz_flag = int(npz_data["box_prompts_flag"])
        # 验证一致性：如果 NPZ 中的 flag 与 count 不一致，以 count 为准并警告
        if (npz_flag == 1 and box_prompts_count == 0) or (npz_flag == 0 and box_prompts_count > 0):
            # 不一致，以 count 为准（已在上面设置）
            pass
        else:
            box_prompts_flag = npz_flag
    
    return {
        "teacher_features": teacher_features,
        "edge_256x256": edge_256x256,
        "fg_map": fg_tensor,
        "bg_map": bg_tensor,
        "edge_map": edge_tensor,
        "box_prompts_xyxy": box_prompts_xyxy,
        "box_prompts_masks_orig": box_prompts_masks_orig,
        "box_prompts_count": box_prompts_count,
        "box_prompts_meta": box_prompts_meta,
        "box_prompts_flag": box_prompts_flag,
        "geometry_color_flag": geometry_color_flag,
        "has_bbox": has_bbox,
        "num_bboxes": num_bboxes,
        "image_shape": image_shape,  # 保存原图尺寸 [H, W, C] 或 [H, W]，如果不存在则为 None
    }


class NPZWithImageIdDataset(Dataset):
    def __init__(
        self,
        features_dir: str,
        images_dir: str | None = None,
        input_size: int = 1024,
        verify_npz: bool = False,
        bad_npz_cache: str | None = None,
        verbose: bool = True,
        assume_clean: bool = False,
    ):
        self.features_dir = Path(features_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.input_size = input_size
        self.verify_npz = verify_npz
        self._verbose = verbose
        self.assume_clean = assume_clean
        if self.images_dir is None:
            raise ValueError("images_dir 必须指定，并指向预处理后的1024图像目录")
        npz_files = sorted(self.features_dir.glob("*_features.npz"))

        self.bad_npz_cache_path = Path(bad_npz_cache) if bad_npz_cache else self.features_dir / "_bad_npz_cache.txt"
        self.corrupted_npz: set[str] = set()

        if self.verify_npz:
            self.corrupted_npz = self._load_corrupted_cache()
            if not self.assume_clean and not self.corrupted_npz and not self.bad_npz_cache_path.exists():
                self.corrupted_npz = self._scan_and_cache_corrupted(npz_files)
            if self.corrupted_npz:
                before = len(npz_files)
                npz_files = [p for p in npz_files if p.name not in self.corrupted_npz]
                removed = before - len(npz_files)
                if self._verbose and removed > 0:
                    logger.info("[NPZ] Skip %s corrupted feature files (cache: %s)", removed, str(self.bad_npz_cache_path))
        elif self.assume_clean and self._verbose:
            logger.info("[NPZ] assume-clean 启用，未执行NPZ完整性扫描，使用全部 %s 个样本。", len(npz_files))

        self.valid_files: List[Path] = npz_files
        self._image_path_cache: dict[str, Path] = {}

    @contextlib.contextmanager
    def _open_npz(self, npz_file: Path):
        # 直接使用 allow_pickle=True，因为这些 NPZ 文件包含对象数组（如 box_prompts_meta）
        data = np.load(npz_file, allow_pickle=True)
        try:
            yield data
        finally:
            data.close()

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        npz_file = self.valid_files[idx]
        image_id = npz_file.stem.replace("_features", "")
        try:
            with self._open_npz(npz_file) as data:
                # ✅ 使用公共解析函数，消除代码重复
                parsed_data = _parse_npz_data(data, verbose=self._verbose)
        except (BadZipFile, ValueError, OSError) as exc:
            raise CorruptNPZError(str(npz_file)) from exc

        image_tensor = self._load_image_from_dir(image_id)

        return {
            "image": image_tensor,
            "image_id": image_id,
            **parsed_data,  # 展开所有解析后的数据
        }
    
    def mark_corrupt(self, file_path: str):
        file_path = str(file_path)
        name = Path(file_path).name
        if name in self.corrupted_npz:
            return
        self.corrupted_npz.add(name)
        # 移除列表中的文件
        for idx, p in enumerate(self.valid_files):
            if p.name == name:
                del self.valid_files[idx]
                break
        # 追加写入缓存
        try:
            self.bad_npz_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.bad_npz_cache_path.exists():
                existing = {line.strip() for line in self.bad_npz_cache_path.read_text().splitlines() if line.strip()}
                if name in existing:
                    return
            with self.bad_npz_cache_path.open("a", encoding="utf-8") as f:
                f.write(name + "\n")
        except Exception as exc:
            if self._verbose:
                logger.warning("[NPZ] Failed to update bad NPZ cache: %s", exc)

    def _load_corrupted_cache(self) -> set[str]:
        if not self.bad_npz_cache_path.exists():
            return set()
        try:
            lines = [line.strip() for line in self.bad_npz_cache_path.read_text(encoding="utf-8").splitlines()]
            return {line for line in lines if line}
        except Exception as exc:
            if self._verbose:
                logger.warning("[NPZ] Failed to read bad NPZ cache: %s. Will rescan.", exc)
            return set()

    def _scan_and_cache_corrupted(self, files: List[Path]) -> set[str]:
        if self._verbose:
            logger.info("[NPZ] Scanning %s feature NPZ files for integrity ...", len(files))
        bad: set[str] = set()
        total = len(files)
        for idx, path in enumerate(files, start=1):
            try:
                with np.load(path, allow_pickle=True):
                    pass
            except Exception:
                bad.add(path.name)
            if self._verbose and idx % 5000 == 0:
                logger.info("[NPZ]   Checked %s/%s files, bad=%s", idx, total, len(bad))
        try:
            self.bad_npz_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.bad_npz_cache_path.open("w", encoding="utf-8") as f:
                for name in sorted(bad):
                    f.write(name + "\n")
        except Exception as exc:
            if self._verbose:
                logger.warning("[NPZ] Failed to write bad NPZ cache: %s", exc)
        if self._verbose:
            logger.info("[NPZ] Scan finished. Corrupted count=%s (cache saved to %s)", len(bad), self.bad_npz_cache_path)
        return bad

    def _find_image_path(self, image_id: str) -> Path | None:
        if self.images_dir is None or not image_id:
            return None

        cache_hit = self._image_path_cache.get(image_id)
        if cache_hit is not None:
            return cache_hit

        raw = image_id.strip()
        if not raw:
            return None

        stem, ext = os.path.splitext(raw)
        normalized = stem if ext else raw
        if normalized.startswith("sa_"):
            normalized = normalized[3:]

        if not normalized:
            return None

        candidate_path = self.images_dir / f"sa_{normalized}.jpg"
        if not candidate_path.exists():
            return None

        self._image_path_cache[image_id] = candidate_path
        return candidate_path

    def _load_image_from_dir(self, image_id: str) -> torch.Tensor:
        img_path = self._find_image_path(image_id)
        if img_path is None:
            raise FileNotFoundError(f"未在 {self.images_dir} 中找到预处理图像 (id={image_id})")
        try:
            try:
                from torchvision.io import read_image  # type: ignore

                img_tensor = read_image(str(img_path))  # [C, H, W] uint8
                if img_tensor.shape[1] != self.input_size or img_tensor.shape[2] != self.input_size:
                    raise ValueError(
                        f"图像尺寸不匹配: {img_path} -> {tuple(img_tensor.shape[1:])}, "
                        f"期望 {self.input_size}x{self.input_size}."
                    )
                # 保持 uint8，归一化放到 GPU 端完成
                return img_tensor
            except Exception:
                # 回退到 PIL 路径
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    if img.size != (self.input_size, self.input_size):
                        raise ValueError(
                            f"图像尺寸不匹配: {img_path} -> {img.size[::-1]}, 期望 {self.input_size}x{self.input_size}."
                            " 请确保已运行1024预处理脚本。"
                        )
                    img_np = np.asarray(img, dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(f"无法读取图像文件: {img_path}. PIL 报错: {exc}") from exc

        # PIL 分支：将 HWC uint8 转为 CHW uint8，归一化仍在 GPU 上完成
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        return img_tensor


class TarShardNPZDataset(Dataset):
    """
    从 tar shard 文件中读取 NPZ 和图像的 Dataset。
    
    设计目标：
    - 与 NPZWithImageIdDataset 完全兼容的输出格式
    - 从 tar 文件读取，减少文件系统元数据竞争
    - 支持 tar.gz 压缩格式
    """
    def __init__(
        self,
        shard_dir: str,
        input_size: int = 1024,
        verbose: bool = True,
        use_cache: bool = True,
    ):
        self.shard_dir = Path(shard_dir)
        self.input_size = input_size
        self._verbose = verbose
        
        # 扫描所有 shard 文件（支持 .tar, .tar.gz, .tar.xz, .tar.bz2）
        shard_files = []
        for pattern in ["*.tar", "*.tar.gz", "*.tar.xz", "*.tar.bz2"]:
            shard_files.extend(sorted(self.shard_dir.glob(pattern)))
        
        if not shard_files:
            raise RuntimeError(f"在 {shard_dir} 下没有找到 tar shard 文件")
        
        if self._verbose:
            logger.info(f"[TAR SHARD] 找到 {len(shard_files)} 个 shard 文件")
        
        # 计算shard文件签名（用于缓存验证）
        def _compute_shard_signature(shard_path: Path) -> dict:
            stat = shard_path.stat()
            return {
                "path": str(shard_path.name),  # 只保存文件名，避免路径变化
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        
        shard_signatures = {str(f): _compute_shard_signature(f) for f in shard_files}
        signature_hash = hashlib.md5(
            json.dumps(sorted(shard_signatures.items()), sort_keys=True).encode()
        ).hexdigest()
        
        # 缓存文件路径
        cache_file = self.shard_dir / f".shard_index_cache_{signature_hash[:8]}.json"
        
        # 尝试加载缓存
        samples = None
        if use_cache and cache_file.exists():
            try:
                if self._verbose:
                    logger.info(f"[TAR SHARD] 尝试加载缓存: {cache_file.name}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                # 验证签名是否匹配
                if cache_data.get("signature_hash") == signature_hash:
                    samples = [
                        (Path(s[0]), s[1], s[2])  # 恢复Path对象
                        for s in cache_data["samples"]
                    ]
                    if self._verbose:
                        logger.info(f"[TAR SHARD] 缓存加载成功，共 {len(samples)} 个样本")
                else:
                    if self._verbose:
                        logger.info(f"[TAR SHARD] 缓存签名不匹配，重新扫描")
            except Exception as e:
                if self._verbose:
                    logger.warning(f"[TAR SHARD] 加载缓存失败: {e}，重新扫描")
        
        # 如果缓存无效，重新扫描
        if samples is None:
            samples = []
            image_id_to_idx = {}
            
            for shard_path in tqdm(shard_files, desc="扫描 shard", disable=not verbose):
                try:
                    with tarfile.open(shard_path, 'r:*') as tar:
                        # 获取所有成员
                        members = tar.getmembers()
                        
                        # 建立 basename -> arcname 映射，避免反复构建 set
                        base_to_name = {os.path.basename(m.name): m.name for m in members}
                        
                        # 找到所有 *_features.npz 文件的 basename
                        npz_bases = [
                            os.path.basename(m.name)
                            for m in members
                            if m.name.endswith('_features.npz')
                        ]
                        
                        for npz_base in npz_bases:
                            # npz_base 形如 "sa_xxx_features.npz"
                            image_id = npz_base.replace('_features.npz', '')  # "sa_xxx"
                            
                            # 对应 jpg 的 basename
                            img_base = f"{image_id}.jpg"  # "sa_xxx.jpg"
                            
                            if img_base in base_to_name:
                                npz_arcname = base_to_name[npz_base]  # 可能带子目录，但通过 basename 匹配
                                img_arcname = base_to_name[img_base]
                                idx = len(samples)
                                samples.append((shard_path, npz_arcname, img_arcname))
                                image_id_to_idx[image_id] = idx
                except Exception as e:
                    if self._verbose:
                        logger.warning(f"[TAR SHARD] 扫描 {shard_path} 时出错: {e}")
            
            # 保存缓存
            if use_cache:
                try:
                    cache_data = {
                        "signature_hash": signature_hash,
                        "samples": [
                            (str(s[0]), s[1], s[2])  # Path转为字符串
                            for s in samples
                        ],
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    if self._verbose:
                        logger.info(f"[TAR SHARD] 缓存已保存: {cache_file.name}")
                except Exception as e:
                    if self._verbose:
                        logger.warning(f"[TAR SHARD] 保存缓存失败: {e}")
        
        self.samples = samples
        
        # 重建 image_id_to_idx（从 npz_arcname 提取 image_id）
        self.image_id_to_idx = {}
        for idx, (_, npz_arcname, _) in enumerate(self.samples):
            npz_base = os.path.basename(npz_arcname)
            image_id = npz_base.replace('_features.npz', '')
            self.image_id_to_idx[image_id] = idx
        
        if not self.samples:
            raise RuntimeError(f"在 shard 文件中没有找到有效的样本对")
        
        if self._verbose:
            logger.info(f"[TAR SHARD] 共 {len(self.samples)} 个有效样本")
        
        self._shard_cache: dict[Path, tarfile.TarFile] = {}  # 缓存打开的 tar 文件
    
    def _get_tar(self, shard_path: Path) -> tarfile.TarFile:
        """获取 tar 文件对象（带缓存）"""
        if shard_path not in self._shard_cache:
            self._shard_cache[shard_path] = tarfile.open(shard_path, 'r:*')
        return self._shard_cache[shard_path]
    
    def _load_from_tar(self, tar: tarfile.TarFile, arcname: str) -> bytes:
        """从 tar 中读取文件内容为 bytes"""
        member = tar.getmember(arcname)
        fileobj = tar.extractfile(member)
        if fileobj is None:
            raise ValueError(f"无法从 tar 中提取 {arcname}")
        try:
            return fileobj.read()
        finally:
            fileobj.close()
    
    @contextlib.contextmanager
    def _open_npz_from_bytes(self, npz_bytes: bytes):
        """从 bytes 打开 NPZ 文件"""
        import io
        # 直接使用 allow_pickle=True，因为这些 NPZ 文件包含对象数组（如 box_prompts_meta）
        data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
        try:
            yield data
        finally:
            data.close()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        shard_path, npz_arcname, img_arcname = self.samples[idx]
        # 从 npz_arcname 的 basename 提取 image_id（兼容带子目录的情况）
        npz_base = os.path.basename(npz_arcname)
        image_id = npz_base.replace('_features.npz', '')
        
        # 获取 tar 文件
        tar = self._get_tar(shard_path)
        
        # 读取 NPZ 数据
        npz_bytes = self._load_from_tar(tar, npz_arcname)
        
        try:
            with self._open_npz_from_bytes(npz_bytes) as data:
                # 以下逻辑与 NPZWithImageIdDataset 完全相同
                if "P4_S16" in data:
                    teacher_np = data["P4_S16"]
                else:
                    teacher_np = data["IMAGE_EMB_S16"]
                if teacher_np.dtype != np.float32:
                    teacher_np = teacher_np.astype(np.float32, copy=False)
                teacher_features = torch.from_numpy(teacher_np).float().squeeze(0).contiguous()
                Hf, Wf = int(teacher_features.shape[-2]), int(teacher_features.shape[-1])
                
                edge_np = data["edge_256x256"]
                if edge_np.dtype != np.float32:
                    edge_np = edge_np.astype(np.float32, copy=False)
                edge_256x256 = torch.from_numpy(edge_np).float()
                
                fg_key = f"fg_map_{Hf}x{Wf}"
                bg_key = f"bg_map_{Hf}x{Wf}"
                edge_key = f"edge_map_{Hf}x{Wf}"
                
                if fg_key in data:
                    fg_arr = data[fg_key]
                    if fg_arr.dtype != np.float32:
                        fg_arr = fg_arr.astype(np.float32, copy=False)
                    fg_tensor = torch.from_numpy(fg_arr).unsqueeze(0)
                else:
                    fg_tensor = torch.zeros(1, Hf, Wf, dtype=torch.float32)
                
                if bg_key in data:
                    bg_arr = data[bg_key]
                    if bg_arr.dtype != np.float32:
                        bg_arr = bg_arr.astype(np.float32, copy=False)
                    bg_tensor = torch.from_numpy(bg_arr).unsqueeze(0)
                else:
                    bg_tensor = torch.full((1, Hf, Wf), 1.0 / max(Hf * Wf, 1), dtype=torch.float32)
                
                edge_tensor = None
                if edge_key in data:
                    edge_arr = data[edge_key]
                    if edge_arr.dtype != np.float32:
                        edge_arr = edge_arr.astype(np.float32, copy=False)
                    edge_tensor = torch.from_numpy(edge_arr).unsqueeze(0)
                elif "edge_256x256" in data and cv2 is not None:
                    # 如果 edge_map_{Hf}x{Wf} 不存在，从 edge_256x256 下采样生成
                    edge_256 = data["edge_256x256"]  # (256, 256) uint8 {0, 1}
                    # 转换为 float32
                    edge_float = edge_256.astype(np.float32)
                    # 使用 cv2.resize + INTER_AREA 下采样
                    edge_resized = cv2.resize(
                        edge_float,
                        (Wf, Hf),  # 注意：cv2.resize 是 (width, height)
                        interpolation=cv2.INTER_AREA
                    )
                    # 阈值化：> 0 则为 1
                    edge_binary = (edge_resized > 0).astype(np.float32)
                    edge_tensor = torch.from_numpy(edge_binary).unsqueeze(0)
                
                box_prompts_xyxy = torch.empty(0, 4, dtype=torch.float32)
                box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
                box_prompts_count = 0
                box_prompts_meta: List | List[dict] = []
                box_prompts_flag = 0

                if "box_prompts_xyxy" in data and "box_prompts_masks_orig" in data:
                    # 已有预处理好的 box_prompts
                    bx_arr = data["box_prompts_xyxy"]
                    if bx_arr.dtype != np.float32:
                        bx_arr = bx_arr.astype(np.float32, copy=False)
                    bx = np.array(bx_arr, copy=False)
                    if bx.ndim == 1:
                        bx = bx.reshape(1, 4)
                    box_prompts_xyxy = torch.from_numpy(bx)

                    bm_arr = data["box_prompts_masks_orig"]
                    if bm_arr.dtype != np.float32:
                        bm_arr = bm_arr.astype(np.float32, copy=False)
                    bm = np.array(bm_arr, copy=False)
                    if bm.ndim == 2:
                        bm = bm.reshape(1, *bm.shape)
                    box_prompts_masks_orig = torch.from_numpy(bm)

                    # 优先使用 NPZ 中的 box_prompts_count，否则从 box_prompts_xyxy 的形状推断
                    if "box_prompts_count" in data:
                        box_prompts_count = int(data["box_prompts_count"])
                    else:
                        # 从 box_prompts_xyxy 的形状推断（如果为空tensor则为0）
                        box_prompts_count = box_prompts_xyxy.shape[0] if box_prompts_xyxy.numel() > 0 else 0
                    
                    if "box_prompts_meta" in data:
                        box_prompts_meta = data["box_prompts_meta"].tolist() if hasattr(data["box_prompts_meta"], "tolist") else data["box_prompts_meta"]
                    
                    # 根据 box_prompts_count 判断是否有框
                    box_prompts_flag = 1 if box_prompts_count > 0 else 0
                elif "bboxes" in data and "masks" in data and "image_shape" in data:
                    # 从原始 bboxes 和 masks 转换（适配1024x1024输入和256x256 mask）
                    # 注意：NPZ中的bboxes是XYWH格式 [x, y, w, h]，需要转换为XYXY格式 [x0, y0, x1, y1]
                    bboxes_orig = data["bboxes"]  # [N, 4] XYWH格式，原图坐标系
                    masks_orig = data["masks"]  # [N, H_orig, W_orig] 或 object array
                    image_shape = data["image_shape"]  # [H, W, C] 或 [H, W]
                    
                    # 获取原图尺寸（用于mask缩放）
                    if image_shape.ndim >= 2:
                        orig_h, orig_w = int(image_shape[0]), int(image_shape[1])
                    else:
                        orig_h, orig_w = 1024, 1024  # 默认值
                    
                    # 转换bboxes：从XYWH格式转换为XYXY格式（原图坐标系，不缩放）
                    # SAM2的prompt_encoder接受原图尺寸的框，内部会根据input_image_size处理
                    if bboxes_orig.size > 0:
                        bboxes_np = np.array(bboxes_orig, dtype=np.float32)
                        if bboxes_np.ndim == 1:
                            bboxes_np = bboxes_np.reshape(1, 4)
                        
                        # 转换XYWH -> XYXY: [x, y, w, h] -> [x0, y0, x1, y1]
                        # x0 = x, y0 = y, x1 = x + w, y1 = y + h
                        bboxes_xyxy = np.zeros_like(bboxes_np)
                        bboxes_xyxy[:, 0] = bboxes_np[:, 0]  # x0 = x
                        bboxes_xyxy[:, 1] = bboxes_np[:, 1]  # y0 = y
                        bboxes_xyxy[:, 2] = bboxes_np[:, 0] + bboxes_np[:, 2]  # x1 = x + w
                        bboxes_xyxy[:, 3] = bboxes_np[:, 1] + bboxes_np[:, 3]  # y1 = y + h
                        
                        box_prompts_xyxy = torch.from_numpy(bboxes_xyxy).float()
                        box_prompts_count = bboxes_xyxy.shape[0]
                    else:
                        box_prompts_xyxy = torch.empty(0, 4, dtype=torch.float32)
                        box_prompts_count = 0
                    
                    # 转换 masks：从原图尺寸缩放到256x256
                    if masks_orig.size > 0 and box_prompts_count > 0:
                        # 取第一个mask（假设每个样本只有一个mask）
                        mask_orig = masks_orig[0] if hasattr(masks_orig, '__len__') and len(masks_orig) > 0 else masks_orig
                        if isinstance(mask_orig, np.ndarray):
                            # 转换为float32并缩放到256x256
                            if mask_orig.dtype != np.float32:
                                mask_orig = mask_orig.astype(np.float32)
                            
                            # 使用cv2.resize缩放（如果可用）
                            if cv2 is not None:
                                mask_256 = cv2.resize(mask_orig, (256, 256), interpolation=cv2.INTER_AREA)
                                # 二值化：>0则为1.0
                                mask_256 = (mask_256 > 0).astype(np.float32)
                            else:
                                # 回退到torch的插值（更可靠，无需额外依赖）
                                mask_tensor = torch.from_numpy(mask_orig).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                                mask_256_tensor = F.interpolate(mask_tensor, size=(256, 256), mode='nearest', align_corners=None)
                                mask_256 = (mask_256_tensor.squeeze().numpy() > 0).astype(np.float32)
                            
                            box_prompts_masks_orig = torch.from_numpy(mask_256).float().unsqueeze(0)  # [1, 256, 256]
                        else:
                            box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
                            box_prompts_count = 0
                    else:
                        box_prompts_masks_orig = torch.empty(0, 256, 256, dtype=torch.float32)
                        box_prompts_count = 0
                    
                    box_prompts_flag = 1 if box_prompts_count > 0 else 0
                    box_prompts_meta = []

                has_bbox_arr = data.get("has_bbox", np.array(False))
                has_bbox = bool(np.array(has_bbox_arr).item()) if isinstance(has_bbox_arr, np.ndarray) else bool(has_bbox_arr)

                if "num_bboxes" in data:
                    num_bboxes = int(np.array(data["num_bboxes"]).item())
                else:
                    num_bboxes = int(box_prompts_count)

                # bboxes 和 masks 已移除（未在训练中使用）

                geom_flag_arr = data.get("geometry_color_flag", np.array(0, dtype=np.uint8))
                geometry_color_flag = int(np.array(geom_flag_arr).item()) if isinstance(geom_flag_arr, np.ndarray) else int(geom_flag_arr)
                
                # 读取图像数据
                img_bytes = self._load_from_tar(tar, img_arcname)
                img_tensor = self._load_image_from_bytes(img_bytes)
                
                result = {
                    "teacher_features": teacher_features,
                    "edge_256x256": edge_256x256,
                    "image": img_tensor,
                    "image_id": image_id,
                    "fg_map": fg_tensor,
                    "bg_map": bg_tensor,
                    "edge_map": edge_tensor if edge_tensor is not None else torch.empty(0),
                    "box_prompts_xyxy": box_prompts_xyxy,
                    "box_prompts_masks_orig": box_prompts_masks_orig,
                    "box_prompts_count": box_prompts_count,
                    "box_prompts_meta": box_prompts_meta,
                    "box_prompts_flag": box_prompts_flag,
                    "geometry_color_flag": geometry_color_flag,
                    "has_bbox": has_bbox,
                    "num_bboxes": num_bboxes,
                    "image_shape": data.get("image_shape"),  # 保存原图尺寸 [H, W, C] 或 [H, W]
                }
                
                return result
        except Exception as e:
            raise RuntimeError(f"处理样本 {idx} (shard={shard_path}, npz={npz_arcname}) 时出错: {e}") from e
    
    def _load_image_from_bytes(self, img_bytes: bytes) -> torch.Tensor:
        """从 bytes 加载图像"""
        try:
            from torchvision.io import read_image
            import io
            import tempfile
            
            # read_image 需要文件路径或文件对象，我们创建一个临时文件对象
            img_file = io.BytesIO(img_bytes)
            # 使用 PIL 读取后再用 torchvision 处理（更兼容）
            from PIL import Image
            img = Image.open(img_file)
            img = img.convert("RGB")
            
            # 转换为 numpy 数组再转 torch tensor
            img_np = np.asarray(img, dtype=np.uint8)
            if img_np.shape[2] == 3:
                # HWC -> CHW
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().clone().clone()
            else:
                raise ValueError(f"意外的图像通道数: {img_np.shape}")
            
            if img_tensor.shape[1] != self.input_size or img_tensor.shape[2] != self.input_size:
                raise ValueError(
                    f"图像尺寸不匹配: {img_tensor.shape[1:]} != ({self.input_size}, {self.input_size})"
                )
            
            return img_tensor
        except Exception as e:
            # 回退到 PIL（更简单直接）
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
            if img.size != (self.input_size, self.input_size):
                raise ValueError(
                    f"图像尺寸不匹配: {img.size} != ({self.input_size}, {self.input_size})"
                )
            img_np = np.asarray(img, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().clone()
            return img_tensor
    
    def __del__(self):
        """清理时关闭所有打开的 tar 文件"""
        for tar in self._shard_cache.values():
            try:
                tar.close()
            except Exception:
                pass


class StreamingTarDataset(IterableDataset):
    """
    【极速 IO 纯净版】

    严格顺序读取，无 Shuffle，无 Buffer 等待，直通 GPU。

    策略：HDD 顺序读取整个 Tar -> (可选)复制到 RAM -> Worker 解析 -> 直出

    """
    def __init__(
        self,
        shard_dir: str,
        input_size: int = 1024,
        shuffle_buffer_size: int = 0, # 保留接口兼容，但内部强制忽略
        verbose: bool = False,
        use_ram_cache: bool = True, 
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.input_size = input_size
        self.verbose = verbose
        self.use_ram_cache = use_ram_cache
        self.shuffle_buffer_size = shuffle_buffer_size  # 保留属性以保持兼容性，但内部不使用
        
        # 1. 扫描并强制排序，确保确定性
        self.shard_files = sorted(
            list(self.shard_dir.glob("*.tar")) + 
            list(self.shard_dir.glob("*.tar.gz"))
        )
        
        if not self.shard_files:
            raise RuntimeError(f"Fatal: No tar files found in {shard_dir}")
        
        if self.verbose:
            logger.info(f"[IO] Found {len(self.shard_files)} shards. (Mode: Strict Sequential, No Shuffle)")

    def _check_ram_space(self, size_needed):
        try:
            if not os.path.exists('/dev/shm'):
                return False
            total, used, free = shutil.disk_usage('/dev/shm')
            # 预留 4GB 安全水位
            return (size_needed + 4 * 1024**3) < free
        except Exception:
            return False

    def _parse_tar_content(self, tar_path):
        process_path = tar_path
        temp_file = None
        fd = None
        
        # === RAM Cache 逻辑 (保持不变，这对 HDD 很重要) ===
        if self.use_ram_cache:
            try:
                file_size = os.path.getsize(tar_path)
                if self._check_ram_space(file_size):
                    fd, temp_path = tempfile.mkstemp(dir='/dev/shm', suffix='.tar')
                    os.close(fd)
                    fd = None 
                    temp_file = temp_path 
                    shutil.copyfile(tar_path, temp_path)
                    process_path = Path(temp_path)
                    
                    if self.verbose:
                        worker_info = torch.utils.data.get_worker_info()
                        wid = worker_info.id if worker_info else 0
                        logger.info(f"[Worker {wid}] Loaded {tar_path.name} to RAM")
            except Exception:
                process_path = tar_path

        # === 读取逻辑 ===
        local_buffer = {} 
        try:
            with tarfile.open(process_path, mode='r|*') as tar:
                for member in tar:
                    if not member.isfile(): continue
                    fname = member.name
                    if '/' in fname: fname = fname.split('/')[-1]
                    
                    if fname.endswith('.npz'):
                        img_id = fname[:-13]
                        type_k = 'npz'
                    elif fname.endswith('.jpg'):
                        img_id = fname[:-4]
                        type_k = 'img'
                    else:
                        continue
                    
                    f_obj = tar.extractfile(member)
                    if f_obj is None: continue
                    try:
                        content = f_obj.read()
                    finally:
                        f_obj.close()
                    
                    if img_id not in local_buffer:
                        local_buffer[img_id] = {type_k: content}
                    else:
                        local_buffer[img_id][type_k] = content
                    
                    # 配对成功立即 Yield，不做任何等待
                    if 'npz' in local_buffer[img_id] and 'img' in local_buffer[img_id]:
                        item = local_buffer.pop(img_id)
                        yield self._process_sample_fast(img_id, item['npz'], item['img'])
                        
        except Exception as e:
            logger.error(f"Error reading tar {process_path}: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def _process_sample_fast(self, image_id, npz_bytes, img_bytes):
        import io
        from torchvision.io import decode_image, ImageReadMode
        
        # 极速解码
        img_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        img_tensor = decode_image(img_buffer, mode=ImageReadMode.RGB)
        
        if img_tensor.shape[1] != self.input_size:
             import torchvision.transforms.functional as F_vis
             img_tensor = F_vis.resize(img_tensor, [self.input_size, self.input_size])
        
        with io.BytesIO(npz_bytes) as f:
            with np.load(f, allow_pickle=True) as data:
                parsed = _parse_npz_data_fast(data)
        
        parsed["image"] = img_tensor
        parsed["image_id"] = image_id
        return parsed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. 严格排序
        all_shards = list(self.shard_files) # self.shard_files 已经在 init 里 sorted 过了
        
        if worker_info is None:
            my_shards = all_shards
        else:
            # 2. 顺序分片：Worker 0 拿第 0, 4, 8... 个文件
            # 这种访问模式对 HDD 最友好，因为文件列表是固定的
            my_shards = all_shards[worker_info.id :: worker_info.num_workers]
        
        # 3. 彻底移除 Shuffle！
        # 直接按顺序一个个读，读完一个 Tar 再读下一个
        for shard_path in my_shards:
            yield from self._parse_tar_content(shard_path)


class DataLoadDiagnostics:
    """数据加载瓶颈诊断工具"""
    def __init__(self, slow_threshold: float = 2.0, fast_threshold: float = 0.5):
        self.data_load_times: list[float] = []
        self.slow_batches: list[tuple[int, float, str]] = []  # (batch_idx, time, reason)
        self.fast_batch_count = 0
        self.slow_threshold = slow_threshold
        self.fast_threshold = fast_threshold
        self.last_slow_batch_idx = -1
        
    def record(self, batch_idx: int, load_time: float) -> str | None:
        """
        记录数据加载时间，检测慢batch并返回原因字符串（如果有）
        
        Returns:
            reason: 如果是慢batch，返回原因字符串；否则返回None
        """
        self.data_load_times.append(load_time)
        
        # 检测慢batch
        if load_time > self.slow_threshold:
            # 判断原因
            if self.fast_batch_count > 10:
                reason = f"prefetch_buffer_exhausted (after {self.fast_batch_count} fast batches)"
            elif load_time > 5.0:
                reason = "very_slow_io (possibly disk bottleneck)"
            elif load_time > 3.0:
                reason = "slow_io (disk may be busy)"
            else:
                reason = "normal_slow_io"
            
            self.slow_batches.append((batch_idx, load_time, reason))
            prev_fast = self.fast_batch_count
            self.fast_batch_count = 0
            
            # 返回原因字符串，用于立即输出
            return f"[SLOW BATCH #{batch_idx}] {load_time:.2f}s - {reason} (was {prev_fast} fast batches before)"
        elif load_time < self.fast_threshold:
            self.fast_batch_count += 1
        else:
            # 中等速度，重置fast计数（因为不是连续的快速batch）
            if self.fast_batch_count > 0:
                self.fast_batch_count = 0
        
        return None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self.data_load_times:
            return {}
        import numpy as np
        times = np.array(self.data_load_times)
        return {
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'p50': float(np.percentile(times, 50)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
            'slow_count': len(self.slow_batches),
            'total_count': len(self.data_load_times),
        }
    
    def get_recent_stats(self, last_n: int) -> dict:
        """获取最近N个batch的统计"""
        if not self.data_load_times or last_n <= 0:
            return {
                'slowest': 0.0,
                'fastest': 0.0,
                'avg': 0.0,
                'num_slow': 0,
            }
        import numpy as np
        recent_times = np.array(self.data_load_times[-last_n:])
        
        # 如果 recent_times 为空，返回默认值
        if len(recent_times) == 0:
            return {
                'slowest': 0.0,
                'fastest': 0.0,
                'avg': 0.0,
                'num_slow': 0,
            }
        
        # 计算最近N个batch中慢batch的数量
        # 找到最近N个batch的起始索引
        total_batches = len(self.data_load_times)
        start_idx = max(0, total_batches - last_n)
        
        # 计算在这个范围内的慢batch数量
        num_slow = sum(1 for batch_idx, _, _ in self.slow_batches 
                      if batch_idx >= start_idx)
        
        return {
            'slowest': float(np.max(recent_times)),
            'fastest': float(np.min(recent_times)),
            'avg': float(np.mean(recent_times)),
            'num_slow': num_slow,
        }


class DistillSingleTester:
    def __init__(self, config: dict, device: torch.device = None) -> None:
        # 延迟导入：在真正需要时才导入模块
        global _RTDETRBackboneBuilder, _OTHER_MODULES
        if _RTDETRBackboneBuilder is None:
            _RTDETRBackboneBuilder = _import_backbones()
        if _OTHER_MODULES is None:
            _OTHER_MODULES = _import_other_modules()
        
        # 将导入的模块绑定到类属性，方便后续使用
        self._rtdetr_backbone_factory = _RTDETRBackboneBuilder
        for key, value in _OTHER_MODULES.items():
            setattr(self, key, value)
        
        self.config = config
        # 如果未指定设备，则根据CUDA可用性自动选择
        if device is None:
            if torch.cuda.is_available():
                # 使用当前设置的CUDA设备
                cuda_idx = torch.cuda.current_device()
                self.device = torch.device(f"cuda:{cuda_idx}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # 【商业级优化】图像归一化层：集成到模型中，避免循环内的Kernel Launch
        # 创建一个简单的归一化模块，可被torch.compile优化
        class ImageNormalize(nn.Module):
            """将uint8图像归一化到[0,1]的float32，可被torch.compile优化"""
            def __init__(self):
                super().__init__()
                self.scale = 1.0 / 255.0
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, 3, H, W] uint8
                return x.float().mul_(self.scale)
        
        self.image_normalize = ImageNormalize().to(self.device)
        
        # backbone
        self.backbone = self._create_backbone().to(self.device).train()

        # 静态通道推断并创建静态适配器/头
        s4_c, s16_c = infer_feature_channels(self.backbone, device=self.device, img_size=1024)
        teacher_c = int(config.get("teacher_channels", 256))

        self.edge_adapter = self.EdgeAdapterStatic(in_channels=s4_c, target_channels=256, target_size=256).to(self.device)
        self.edge_head = self.UniversalEdgeHead(
            core_channels=256,
            output_channels=1,
            head_type=config.get("head_type", "simple"),
            init_p=0.05,
        ).to(self.device)
        # feature_adapter 支持内置插值，target_size 从配置读取（默认64，对应1024输入下的S16特征）
        teacher_spatial_size = config.get("teacher_spatial_size", 64)
        self.feature_adapter = self.SimpleAdapterStatic(
            in_channels=s16_c, 
            out_channels=teacher_c,
            target_size=teacher_spatial_size
        ).to(self.device)
        self._warned_corrupt_npz: set[str] = set()
        
        # 边缘损失函数（支持渐进式边缘掩码和正类重加权）
        self.edge_loss = self.EdgeDistillationLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            enable_edge_mask=False,  # 初始关闭，由enable_edge_mask_progressive控制
            edge_mask_kernel_size=config.get("edge_mask_kernel_size", 3),
            use_pos_weight=config.get("use_pos_weight", False),
        ).to(self.device)

        # SAM2 辨析头配置（必须在初始化 torchmetrics 之前设置）
        self.enable_mask_loss = bool(config.get("enable_mask_loss", True))  # 默认开启辨析任务
        
        # 详细日志控制：默认关闭，可通过配置显式启用
        enable_detailed_logging_explicit = config.get("enable_detailed_mask_logging", None)
        if enable_detailed_logging_explicit is not None:
            # 如果显式指定了，使用指定值
            self.enable_detailed_mask_logging = bool(enable_detailed_logging_explicit)
        else:
            # 默认关闭详细日志，避免日志过多
            self.enable_detailed_mask_logging = False
        
        if self.enable_detailed_mask_logging:
            logger.info("[MASK TASK] 启用详细诊断日志")
        else:
            logger.info("[MASK TASK] 关闭详细诊断日志（默认关闭，可通过 --enable-detailed-mask-logging 启用）")
        
        # 评估指标（使用 torchmetrics 优化，在设备上累积，减少同步）
        self.eval_metrics: MetricCollection | None = None
        self.mask_metrics: BinaryJaccardIndex | None = None
        if TORCHMETRICS_AVAILABLE:
            try:
                self.eval_metrics = MetricCollection({
                    'mse': MeanSquaredError(),
                    'mae': MeanAbsoluteError(),
                    'cosine_sim': CosineSimilarity(reduction='mean')
                }).to(self.device)
                if self.enable_mask_loss:
                    # BinaryJaccardIndex 在 torchmetrics 1.8+ 中不需要 task 参数
                    self.mask_metrics = BinaryJaccardIndex(threshold=0.5).to(self.device)
            except Exception as exc:
                logger.warning("初始化 torchmetrics 失败，回退到手动计算: %s", exc)
                self.eval_metrics = None
                self.mask_metrics = None
        self.mask_loss_weight = float(config.get("mask_loss_weight", 1.0))
        self.mask_head_unfreeze_epoch = int(config.get("mask_head_unfreeze_epoch", 100))
        self._mask_head_frozen = False
        self.mask_loss: EdgeDistillationLoss | None = None
        self.sam_prompt_encoder = None
        self.sam_mask_decoder = None
        self.sam_image_pe = None

        if self.enable_mask_loss:
            sam2_cfg = config.get("sam2_head_config", "sam2.1/sam2.1_hiera_b+.yaml")
            sam2_ckpt = config.get("sam2_head_ckpt", None)
            try:
                pe, md, transforms = self.load_sam2_heads(
                    device=str(self.device),
                    config_file=sam2_cfg,
                    ckpt_path=sam2_ckpt,
                    return_model=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load SAM2 heads for mask loss: {exc}") from exc

            # ✅ 学生版本（可训练，从教师初始化）
            # 不冻结！让学生 Prompt Encoder 和 Mask Decoder 可训练，适应学生的图像编码器特征
            self.sam_prompt_encoder = pe.to(self.device).train()
            self.sam_mask_decoder = md.to(self.device).train()
            # ❌ 删除冻结代码：让学生版本可训练
            # for module in (self.sam_prompt_encoder, self.sam_mask_decoder):
            #     for param in module.parameters():
            #         param.requires_grad_(False)

            with torch.no_grad():
                self.sam_image_pe = self.sam_prompt_encoder.get_dense_pe()
                if self.sam_image_pe.device != self.device:
                    self.sam_image_pe = self.sam_image_pe.to(self.device)

            mask_bce_weight = config.get("mask_bce_weight", 2.0)
            mask_dice_weight = config.get("mask_dice_weight", 1.0)
            mask_use_pos_weight = config.get("mask_use_pos_weight", False)

            self.mask_loss = self.EdgeDistillationLoss(
                bce_weight=mask_bce_weight,
                dice_weight=mask_dice_weight,
                enable_edge_mask=False,
                edge_mask_kernel_size=config.get("edge_mask_kernel_size", 3),
                use_pos_weight=mask_use_pos_weight,
        ).to(self.device)

            if self.mask_head_unfreeze_epoch > 0:
                self._set_mask_head_trainable(False)
                logger.info(
                    "SAM Prompt/Mask Decoder 将在 epoch >= %s 时解冻训练（当前冻结）",
                    self.mask_head_unfreeze_epoch,
                )
            else:
                self._set_mask_head_trainable(True)
        
        # 渐进式边缘掩码配置
        self.enable_edge_mask_progressive = config.get("enable_edge_mask_progressive", False)
        self.edge_mask_start_epoch = config.get("edge_mask_start_epoch", 5)
        self.edge_task_start_epoch = int(config.get("edge_task_start_epoch", 0))
        self.mask_task_start_epoch = int(config.get("mask_task_start_epoch", 0))
        self.current_epoch = 0
        self._edge_task_activation_logged = False
        self._mask_task_activation_logged = False

        # distill loss
        loss_type = config.get("loss_type", "mse")
        if loss_type == "mse":
            self.distill_loss = self.FeatureLoss({"loss_type": "mse", "alpha": config.get("mse_weight", 1.0)}).to(self.device)
        elif loss_type == "fgd":
            self.distill_loss = self.FGDLoss(
                alpha_fg=config.get("fgd_alpha_fg", 0.001),  # 官方默认
                beta_bg=config.get("fgd_beta_bg", 0.0005),  # 官方默认（前景的一半）
                alpha_edge=config.get("fgd_alpha_edge", 0.002),  # 前景的两倍
                gamma_mask=config.get("fgd_gamma_mask", 0.0),
                lambda_rela=config.get("fgd_lambda_rela", 0.0),
                temperature=config.get("fgd_temperature", 1.0),
            ).to(self.device)
        elif loss_type == "fsdlike":
            self.distill_loss = self.FSDLikeLoss(
                weight_fg=config.get("fsd_weight_fg", 1.0),
                weight_bg=config.get("fsd_weight_bg", 0.2),
                temperature=config.get("fsd_temperature", 1.0),
                gamma_mask=config.get("fsd_gamma_mask", 0.0),
                lambda_rela=config.get("fsd_lambda_rela", 0.0),
                gaussian_from_mask=config.get("fsd_gaussian_from_mask", False),
                gaussian_mix=config.get("fsd_gaussian_mix", "max"),
                gaussian_blend_lambda=config.get("fsd_gaussian_blend_lambda", 0.5),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # 【商业级优化】强制torch.compile：在生产环境必须能编译，不能编译就修复代码
        # 但可以通过 --no-compile 参数禁用，用于快速测试和调试
        if hasattr(torch, "compile") and not config.get("no_compile", False):
            logger.info("🚀 [COMPILE] 正在编译所有核心组件 (mode=reduce-overhead)... 这需要几分钟...")
            try:
                # Fail Fast: 如果编译失败，立即崩溃，不要掩盖问题
                self.backbone = torch.compile(self.backbone, mode="reduce-overhead")
                self.edge_adapter = torch.compile(self.edge_adapter, mode="reduce-overhead")
                self.edge_head = torch.compile(self.edge_head, mode="reduce-overhead")
                self.feature_adapter = torch.compile(self.feature_adapter, mode="reduce-overhead")
                # 归一化层也需要编译
                self.image_normalize = torch.compile(self.image_normalize, mode="reduce-overhead")
                
                # 所有Loss都必须支持编译，包括FGD
                logger.info("[COMPILE] 编译 %s 蒸馏损失...", loss_type.upper())
                self.distill_loss = torch.compile(self.distill_loss, mode="reduce-overhead")
                
                if self.mask_loss is not None:
                    logger.info("[COMPILE] 编译 mask_loss...")
                    self.mask_loss = torch.compile(self.mask_loss, mode="reduce-overhead")
                
                logger.info("✅ [COMPILE] 所有组件编译完成！")
            except Exception as exc:
                logger.warning("⚠️  编译模型失败，回退到 Eager 模式: %s", exc)
        else:
            # 如果禁用了编译，打印一条提示
            if config.get("no_compile", False):
                logger.info("⏩ [FAST START] 已禁用 torch.compile，使用 Eager 模式运行（启动快，运行稍慢）")
            elif not hasattr(torch, "compile"):
                logger.warning("⚠️  torch.compile 不可用，使用 Eager 模式运行")

        all_params = (
            list(self.backbone.parameters())
            + list(self.edge_adapter.parameters())
            + list(self.edge_head.parameters())
            + list(self.feature_adapter.parameters())
        )
        
        # ✅ 添加学生的 Prompt Encoder 和 Mask Decoder 参数（如果启用掩码损失）
        if self.enable_mask_loss:
            if self.sam_prompt_encoder is not None:
                all_params += list(self.sam_prompt_encoder.parameters())
            if self.sam_mask_decoder is not None:
                all_params += list(self.sam_mask_decoder.parameters())
        
        optim_lr = config.get("learning_rate", 1e-3)
        # 禁用 Fused AdamW，避免 found_inf 形状不匹配的 bug
        # RuntimeError: output with shape [] doesn't match the broadcast shape [1]
        self.optimizer = optim.AdamW(all_params, lr=optim_lr, fused=False)
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        self.log_interval = max(1, int(config.get("log_interval", 50)))

    def compute_total_loss_on_batch(self, batch: dict) -> tuple[torch.Tensor, dict]:
        self.backbone.eval()
        self.edge_adapter.eval()
        self.edge_head.eval()
        self.feature_adapter.eval()
        with torch.no_grad():
            images_u8 = batch["image"].to(self.device)
            # 【商业级优化】使用模型内的归一化层，避免循环内的Kernel Launch
            images = self.image_normalize(images_u8).contiguous(memory_format=torch.channels_last)
            teacher_features = batch["teacher_features"].to(self.device)
            edge_256x256 = batch["edge_256x256"].to(self.device)

            feats = self.backbone(images)
            s4 = feats[0]
            s16 = feats[2]
            # feature_adapter 已内置插值，无需外部检查
            aligned = self.feature_adapter(s16)
            feat_loss = F.mse_loss(aligned, teacher_features)

            edge_loss_val = torch.zeros((), device=self.device)
            if self._is_edge_task_active():
                aligned_s4 = self.edge_adapter(s4)
                edge_logits = self.edge_head(aligned_s4)
                edge_loss_dict = self.edge_loss(edge_logits, edge_256x256, return_components=True)
                edge_loss_val = edge_loss_dict["total_loss"]
            else:
                edge_loss_dict = {}

            total = self.config.get("feat_weight", 1.0) * feat_loss + self.config.get("edge_weight", 1.0) * edge_loss_val
            comp = {"feat": feat_loss.item(), "edge": edge_loss_val.item(), "total": total.item()}
            return total, comp

    def _create_backbone(self) -> nn.Module:
        if self._rtdetr_backbone_factory is None:
            raise RuntimeError("RT-DETR backbone工厂尚未初始化")

        backbone = self._rtdetr_backbone_factory(self.config)
        resume_path = self.config.get("student_resume")
        if resume_path:
            resume_path = Path(resume_path).expanduser()
            if resume_path.exists():
                try:
                    checkpoint = torch.load(resume_path, map_location="cpu")
                    if isinstance(checkpoint, dict):
                        if "model" in checkpoint:
                            state_dict = checkpoint["model"]
                        elif "ema" in checkpoint:
                            ema_state = checkpoint["ema"]
                            if isinstance(ema_state, dict) and "module" in ema_state:
                                state_dict = ema_state["module"]
                            else:
                                state_dict = ema_state
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint

                    clean_state: dict[str, torch.Tensor] = {}
                    for key, value in state_dict.items():
                        new_key = key[7:] if key.startswith("module.") else key
                        clean_state[new_key] = value

                    load_result = backbone.load_state_dict(clean_state, strict=False)
                    missing = getattr(load_result, "missing_keys", [])
                    unexpected = getattr(load_result, "unexpected_keys", [])
                    logger.info(
                        "[STUDENT] Loaded backbone weights from %s (missing=%s, unexpected=%s)",
                        resume_path,
                        len(missing),
                        len(unexpected),
                    )
                except Exception as exc:
                    logger.warning("[STUDENT] 无法加载 student_resume (%s): %s", resume_path, exc)
            else:
                logger.warning("[STUDENT] student_resume 路径不存在: %s", resume_path)

        return backbone


    # (intentionally no nested method definitions here)

    def _is_edge_task_active(self, epoch: int | None = None) -> bool:
        if epoch is None:
            epoch = self.current_epoch
        return epoch >= self.edge_task_start_epoch

    def _is_mask_task_active(self, epoch: int | None = None) -> bool:
        if epoch is None:
            epoch = self.current_epoch
        return (
            self.enable_mask_loss
            and self.mask_loss is not None
            and epoch >= self.mask_task_start_epoch
        )

    def export_clean_backbone_state(self, prefixes_to_strip: str | tuple[str, ...] = "backbone.") -> dict[str, torch.Tensor]:
        """
        导出去除 wrapper 前缀的 backbone 权重（用于 MMDet 下游加载）
        """
        raw_state = self.backbone.state_dict()
        if isinstance(prefixes_to_strip, str):
            strip_list = (prefixes_to_strip,)
        else:
            strip_list = prefixes_to_strip
        clean_state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            new_key = key
            for prefix in strip_list:
                if prefix and new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            clean_state[new_key] = value
        return clean_state

    def _set_mask_head_trainable(self, enabled: bool) -> None:
        if not self.enable_mask_loss:
            return
        for module in (self.sam_prompt_encoder, self.sam_mask_decoder):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(enabled)
        self._mask_head_frozen = not enabled

    def _maybe_unfreeze_mask_heads(self, epoch: int) -> None:
        if (
            not self.enable_mask_loss
            or self.mask_head_unfreeze_epoch <= 0
            or not self._mask_head_frozen
        ):
            return
        if epoch >= self.mask_head_unfreeze_epoch:
            self._set_mask_head_trainable(True)
            logger.info(
                "[Epoch %s] 解冻 SAM Prompt/Mask Decoder (mask_head_unfreeze_epoch=%s)",
                epoch,
                self.mask_head_unfreeze_epoch,
            )

    def _compute_distill_loss(
        self,
        p3_features: torch.Tensor,
        teacher_features: torch.Tensor,
        image_ids: List[str],
        return_aligned: bool = False,
        fg_map: torch.Tensor | None = None,
        bg_map: torch.Tensor | None = None,
        edge_map: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 静态适配器已内置插值，无需外部检查（简化计算图，利于 torch.compile）
        aligned_features = self.feature_adapter(p3_features)

        loss_type = self.config.get("loss_type", "mse")
        if loss_type == "mse":
            loss_val = self.distill_loss(aligned_features, teacher_features)
            return (loss_val, aligned_features) if return_aligned else loss_val

        Hf, Wf = aligned_features.shape[-2], aligned_features.shape[-1]

        if fg_map is None or bg_map is None:
            raise RuntimeError(
                "FGD/FSD 蒸馏需要预处理好的 fg_map / bg_map，请确保数据预处理阶段已经写入对应张量。"
            )
        fg_map = fg_map.to(aligned_features.device, non_blocking=True)
        bg_map = bg_map.to(aligned_features.device, non_blocking=True)

        enable_edge_boost = self.config.get("enable_edge_boost", False)
        
        # 根据 enable_edge_boost 控制是否使用 edge_map
        if loss_type == "fgd":
            if enable_edge_boost:
                # 启用边缘增强：必须有 edge_map，否则报错
                if edge_map is None:
                    raise RuntimeError(
                        "启用了 enable_edge_boost，但批数据缺少 edge_map，请确保预处理阶段已写入 edge_map_* 张量。"
                    )
                edge_map = edge_map.to(aligned_features.device, non_blocking=True)
            else:
                # 关闭边缘增强：即使有 edge_map 也不使用，设为 None
                edge_map = None

        if loss_type == "fgd":
            loss_val = self.distill_loss(aligned_features, teacher_features, fg_map=fg_map, bg_map=bg_map, edge_map=edge_map)
        elif loss_type == "fsdlike":
            loss_val = self.distill_loss(aligned_features, teacher_features, fg_map=fg_map, bg_map=bg_map)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        return (loss_val, aligned_features) if return_aligned else loss_val

    def _forward_mask_head(
        self,
        aligned_features: torch.Tensor,
        box_prompts_xyxy: torch.Tensor | list | None,
        box_prompts_masks_orig: torch.Tensor | list | None,
        box_prompts_count: torch.Tensor | list | None,
        image_shapes: list | None = None,  # 每个样本的原图尺寸 [H, W, C] 或 [H, W]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        前向传播 mask head，使用假框填充以保持固定 batch size，避免 torch.compile 重编译。
        
        返回:
            pred_masks_logits: [B, 1, 256, 256] 预测的 mask logits
            gt_masks_binary: [B, 1, 256, 256] 二值化的 GT masks
            valid_mask_vec: [B] 有效样本掩码，1.0 表示真实样本，0.0 表示假框样本（用于 loss 加权）
        """
        if (self.sam_prompt_encoder is None or self.sam_mask_decoder is None):
            if self.enable_detailed_mask_logging:
                logger.warning("[MASK] _forward_mask_head: SAM2 heads not initialized (sam_prompt_encoder or sam_mask_decoder is None)")
            return None, None, None
        
        if box_prompts_xyxy is None or box_prompts_masks_orig is None:
            if self.enable_detailed_mask_logging:
                logger.warning("[MASK] _forward_mask_head: Missing box prompts (box_prompts_xyxy or box_prompts_masks_orig is None)")
            return None, None, None

        # 1. 构造固定的 Batch 输入（不再过滤，而是用假框填充）
        batch_size = aligned_features.shape[0]
        device = aligned_features.device

        # 准备容器：构造 shape 为 [B, 1, 4] 的 tensor
        batched_boxes = torch.zeros((batch_size, 1, 4), device=device, dtype=torch.float32)
        # 记录哪些样本是真实有效的 (用于 loss 计算)
        valid_mask_vec = torch.zeros((batch_size,), device=device, dtype=torch.float32)

        # 解析 count
        if isinstance(box_prompts_count, torch.Tensor):
            counts = box_prompts_count.tolist()
        else:
            counts = [int(c) for c in box_prompts_count]

        gt_masks_list = []  # 用于存 GT

        for i in range(batch_size):
            cnt = counts[i]
            if cnt > 0:
                # === 有真实框 ===
                box = box_prompts_xyxy[i]
                if isinstance(box, torch.Tensor):
                    box = box.to(device)
                else:
                    box = torch.tensor(box, device=device)

                # 确保 box 是 [1, 4] 形状
                if box.ndim == 1:
                    box = box.unsqueeze(0)  # [4] -> [1, 4]
                elif box.ndim == 2 and box.shape[0] > 1:
                    # 如果有多于1个框，只取第一个
                    box = box[0:1]
                
                batched_boxes[i] = box
                valid_mask_vec[i] = 1.0  # 标记为有效

                # 处理 GT Mask
                mask_obj = box_prompts_masks_orig[i]
                if isinstance(mask_obj, np.ndarray):
                    if mask_obj.dtype == object and mask_obj.size > 0:
                        # object array，取出内容
                        mask_np = mask_obj.item(0)
                        m = torch.from_numpy(mask_np.astype(np.float32))
                    else:
                        m = torch.from_numpy(mask_obj.astype(np.float32))
                elif isinstance(mask_obj, torch.Tensor):
                    m = mask_obj
                else:
                    # 异常情况补全
                    m = torch.zeros((256, 256), dtype=torch.float32)

                if m.ndim == 2:
                    m = m.unsqueeze(0)  # [H, W] -> [1, H, W]
                gt_masks_list.append(m.to(device))

            else:
                # === 无框 (0个) ===
                # 填入 Dummy Box [0, 0, 1, 1] 防止 SAM 报错 NaN
                # 这是左上角 1x1 像素的小框，对特征干扰最小
                batched_boxes[i, 0, :] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
                valid_mask_vec[i] = 0.0  # 标记为无效，Loss权重为 0

                # GT Mask 也补一个全黑的，占位
                gt_masks_list.append(torch.zeros((1, 256, 256), device=device))

        # 2. 此时 batched_boxes 永远是 [B, 1, 4]，形状固定！
        # 送入 SAM
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=batched_boxes,  # [B, 1, 4]
            masks=None,
        )

        image_pe = self.sam_image_pe if self.sam_image_pe is not None else self.sam_prompt_encoder.get_dense_pe()

        # 整个 Batch 一起预测
        pred_masks_logits, pred_iou, _, _ = self.sam_mask_decoder(
            image_embeddings=aligned_features,  # [B, C, H, W]
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
        )  # pred_masks_logits: [B, 1, 256, 256]

        # 3. 处理 GT Masks
        # 堆叠并下采样
        gt_masks_stacked = torch.stack(gt_masks_list, dim=0).to(device)  # [B, 1, H_orig, W_orig]
        
        # 如果原始 mask 不是 256x256，需要 interpolate
        if gt_masks_stacked.shape[-1] != 256 or gt_masks_stacked.shape[-2] != 256:
            gt_masks_256 = F.interpolate(gt_masks_stacked, size=(256, 256), mode='area')
        else:
            gt_masks_256 = gt_masks_stacked

        # 二值化
        gt_masks_256_binary = (gt_masks_256 > 0.5).float()

        # 4. 返回 valid_mask_vec 给 Loss 函数
        # 这样我们在算 Loss 时可以把 dummy 样本的 Loss 乘 0
        return pred_masks_logits, gt_masks_256_binary, valid_mask_vec
        
    def train_epoch(self, loader: DataLoader, epoch: int, external_total_batches: int | None = None):
        """
        Train one epoch.
        
        Args:
            loader (DataLoader): The data loader for the training set.
            epoch (int): The current epoch number.
            external_total_batches (int | None): If provided, use this as the total number of batches for tqdm.
        """
        self.backbone.train()
        self.feature_adapter.train()
        if self.edge_adapter is not None:
            self.edge_adapter.train()
        if self.edge_head is not None:
            self.edge_head.train()
        if self.enable_mask_loss:
            if self.sam_prompt_encoder is not None:
                if self._mask_head_frozen:
                    self.sam_prompt_encoder.eval()
                else:
                    self.sam_prompt_encoder.train()
            if self.sam_mask_decoder is not None:
                if self._mask_head_frozen:
                    self.sam_mask_decoder.eval()
                else:
                    self.sam_mask_decoder.train()
            
        # 根据epoch决定是否解冻mask head
        self._maybe_unfreeze_mask_heads(epoch)
        
        self.current_epoch = epoch
        
        # 记录 mask task 激活状态（每个 epoch 开始时打印一次）
        if self._is_mask_task_active(epoch) and not self._mask_task_activation_logged:
            logger.info(f"[MASK TASK] ✅ Mask task activated at epoch {epoch+1} (start_epoch={self.mask_task_start_epoch})")
            self._mask_task_activation_logged = True
        elif not self._is_mask_task_active(epoch) and epoch == self.mask_task_start_epoch - 1:
            logger.info(f"[MASK TASK] ⏳ Mask task will activate at next epoch (epoch {epoch+2})")
        
        # 准备 amp (使用类中已有的 scaler)
        amp_enabled = torch.cuda.is_available()
        
        # 准备 tqdm
        num_batches = len(loader) if external_total_batches is None else external_total_batches
        total_epochs = self.config.get("epochs", "?")
        progress_bar = tqdm(
            enumerate(loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            ncols=120,
        )
        
        # 数据加载诊断
        load_diagnostics = DataLoadDiagnostics(slow_threshold=2.0, fast_threshold=0.5)
        
        # 计时器
        interval_timer = StageTimer(self.device, stages=['data', 'backbone', 'loss', 'backward', 'step', 'metrics'])
        
        batch_times: List[float] = []
        data_load_times: List[float] = []
        
        total_loss_epoch = 0.0
        distill_loss_epoch = 0.0
        edge_loss_epoch = 0.0
        mask_loss_epoch = 0.0
        
        # 学习率记录
        lr_values = []
        
        # DEBUG: 打印前几个batch的详细信息
        debug_steps = 0
        debug_config = self.config.get("debug", {})
        MAX_DEBUG_STEPS = debug_config.get("initial_batch_logs", 0) if isinstance(debug_config, dict) else 0
        
        t_data_start = time.time()

        for batch_idx, batch in progress_bar:
            interval_timer.start('data')
            
            # 停止上一个batch的数据加载计时
            if batch_idx > 0:
                data_load_times.append(time.time() - t_data_start)
                load_time_msg = load_diagnostics.record(batch_idx, data_load_times[-1])
                if load_time_msg:
                    logger.warning(load_time_msg)

            # 数据预取到GPU (如果使用 prefetcher)
            if not isinstance(loader.dataset, IterableDataset):
                try:
                    images = batch["image"].to(self.device, non_blocking=True)
                    teacher_features = batch["teacher_features"].to(self.device, non_blocking=True)
                    # 图像归一化：将 uint8 [0, 255] 转换为 float32 [0, 1]
                    if images.dtype == torch.uint8:
                        images = images.float().div(255.0)
                except Exception as e:
                    logger.error(f"Batch {batch_idx}: Error moving batch to device: {e}")
                    # 尝试打印 batch 内容以诊断
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.error(f"  key='{k}', shape={v.shape}, dtype={v.dtype}")
                        else:
                            logger.error(f"  key='{k}', type={type(v)}")
                    continue
            else:
                # IterableDataset (tar_shard 模式) 也需要移动到设备
                images = batch["image"].to(self.device, non_blocking=True)
                teacher_features = batch["teacher_features"].to(self.device, non_blocking=True)
            
            # 图像归一化：将 uint8 [0, 255] 转换为 float32 [0, 1]
            if images.dtype == torch.uint8:
                images = images.float().div(255.0)
            
            interval_timer.stop('data')

            image_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [*batch["image_id"]]
            box_prompts_xyxy = batch.get("box_prompts_xyxy")
            box_prompts_masks_orig = batch.get("box_prompts_masks_orig")
            box_prompts_count = batch.get("box_prompts_count")
            # 获取每个样本的原图尺寸
            image_shapes = batch.get("image_shape")  # 每个样本的原图尺寸
            if image_shapes is not None and not isinstance(image_shapes, list):
                # 如果是 tensor 或其他类型，转换为列表
                if isinstance(image_shapes, torch.Tensor):
                    image_shapes = [image_shapes[i].cpu().numpy() if image_shapes[i] is not None else None for i in range(len(image_shapes))]
                else:
                    image_shapes = [image_shapes] if image_shapes is not None else None
            fg_map = batch.get("fg_map")
            bg_map = batch.get("bg_map")
            edge_map = batch.get("edge_map")

            if isinstance(fg_map, torch.Tensor) and fg_map.device != self.device:
                fg_map = fg_map.to(self.device, non_blocking=True)
            if isinstance(bg_map, torch.Tensor) and bg_map.device != self.device:
                bg_map = bg_map.to(self.device, non_blocking=True)
            if isinstance(edge_map, torch.Tensor):
                if edge_map.numel() == 0:
                    edge_map = None
                elif edge_map.device != self.device:
                    edge_map = edge_map.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(enabled=amp_enabled):
                # [DEBUG] 3. 开始 Backbone 前向 (这里通常是编译卡顿点)
                if debug_steps < MAX_DEBUG_STEPS:
                    logger.info(f"[DEBUG] Batch {debug_steps}: 开始 Backbone Forward (如卡顿，则是在编译Backbone)...")
                t_bb_start = time.time()
                
                handle = interval_timer.start('backbone')
                features = self.backbone(images)
                interval_timer.stop('backbone', handle)
                
                # [DEBUG] 4. Backbone 结束
                if debug_steps < MAX_DEBUG_STEPS:
                    bb_elapsed = time.time() - t_bb_start
                    logger.info(f"[DEBUG] Batch {debug_steps}: Backbone 完成 (耗时 {bb_elapsed:.4f}s)")
                s4_features = features[0]
                s16_features = features[2]
                
                # [DEBUG] 5. 开始 Loss 计算 (Loss如果也被编译了，这里也会卡)
                if debug_steps < MAX_DEBUG_STEPS:
                    logger.info(f"[DEBUG] Batch {debug_steps}: 开始 Loss 计算...")
                t_loss_start = time.time()
                
                handle = interval_timer.start('loss')
                
                # 总损失和损失字典
                total_loss = torch.tensor(0.0, device=self.device)
                # 初始化 loss_dict，确保所有键都存在，避免 KeyError
                loss_dict = {
                    'distill': 0.0,
                    'edge': 0.0,
                    'mask': 0.0,
                }

                # 1. 蒸馏损失 (基础损失)
                distill_loss = self._compute_distill_loss(
                    p3_features=s16_features,
                    teacher_features=teacher_features,
                    image_ids=image_ids,
                    return_aligned=False, # train时不需要返回对齐特征
                    fg_map=fg_map,
                    bg_map=bg_map,
                    edge_map=edge_map,
                )
                total_loss += distill_loss
                loss_dict['distill'] = distill_loss.item()
                distill_loss_epoch += loss_dict['distill']

                # 2. 边缘任务损失 (如果启用)
                if self._is_edge_task_active(epoch):
                    edge_gt = batch["edge_256x256"].to(self.device, non_blocking=True)
                    s4_features = features[0]
                    aligned_s4 = self.edge_adapter(s4_features)
                    edge_pred_logits = self.edge_head(aligned_s4)
                    
                    edge_loss_dict = self.edge_loss(edge_pred_logits, edge_gt, return_components=True)
                    edge_loss = edge_loss_dict["total_loss"]
                    total_loss += edge_loss * self.config.get("edge_weight", 1.0)
                    loss_dict['edge'] = edge_loss.item()
                    edge_loss_epoch += loss_dict['edge']
                # 如果边缘任务未激活，loss_dict['edge'] 保持为 0.0（已在初始化时设置）
                
                # 3. Mask辨析任务损失 (如果启用)
                if self._is_mask_task_active(epoch):
                    # 获取对齐后的特征，准备送入 mask head
                    aligned_s16_for_mask = self.feature_adapter(s16_features)

                    pred_masks_logits, gt_masks_binary, valid_mask_vec = self._forward_mask_head(
                        aligned_features=aligned_s16_for_mask,
                        box_prompts_xyxy=box_prompts_xyxy,
                        box_prompts_masks_orig=box_prompts_masks_orig,
                        box_prompts_count=box_prompts_count,
                        image_shapes=image_shapes,
                    )
                    
                    # 使用 valid_mask_vec 过滤有效样本计算 loss
                    # 这样假框产生的 loss 会被排除，不会影响模型训练
                    if pred_masks_logits is not None and gt_masks_binary is not None and self.mask_loss is not None:
                        # 方案：只对有效样本计算 Loss（不影响模型编译，只影响 Loss 计算图）
                        valid_bool = valid_mask_vec > 0.5
                        if valid_bool.any():
                            valid_logits = pred_masks_logits[valid_bool]
                            valid_gt = gt_masks_binary[valid_bool]
                            mask_loss = self.mask_loss(valid_logits, valid_gt)
                            
                            total_loss += mask_loss * self.mask_loss_weight
                            loss_dict['mask'] = mask_loss.item()
                            mask_loss_epoch += loss_dict['mask']
                            
                            # 详细日志（如果启用）
                            if self.enable_detailed_mask_logging and batch_idx % 10 == 0:
                                num_valid = valid_bool.sum().item()
                                batch_size_actual = valid_mask_vec.shape[0] if valid_mask_vec is not None else len(box_prompts_count) if box_prompts_count is not None else 0
                                logger.info(f"[MASK] Batch {batch_idx}: valid_samples={num_valid}/{batch_size_actual}, mask_loss={loss_dict['mask']:.6f}")
                        else:
                            # 本 batch 全是背景（假框），loss 为 0
                            loss_dict['mask'] = 0.0
                            if self.enable_detailed_mask_logging and batch_idx % 10 == 0:
                                batch_size_actual = valid_mask_vec.shape[0] if valid_mask_vec is not None else len(box_prompts_count) if box_prompts_count is not None else 0
                                logger.info(f"[MASK] Batch {batch_idx}: No valid samples (all dummy boxes), mask_loss=0.0")
                    else:
                        # _forward_mask_head 返回了 None，说明缺少必要组件
                        loss_dict['mask'] = 0.0
                        if self.enable_detailed_mask_logging and batch_idx % 10 == 0:
                            missing = []
                            if pred_masks_logits is None: missing.append("pred_masks_logits")
                            if gt_masks_binary is None: missing.append("gt_masks_binary")
                            if self.mask_loss is None: missing.append("mask_loss")
                            logger.warning(f"[MASK] Batch {batch_idx}: _forward_mask_head returned None or mask_loss is None. Missing: {missing}")
                else:
                    # Mask task 未激活，不计算 loss
                    loss_dict['mask'] = 0.0
                
                interval_timer.stop('loss', handle)

                # [DEBUG] 6. Loss 计算结束
                if debug_steps < MAX_DEBUG_STEPS:
                    loss_elapsed = time.time() - t_loss_start
                    logger.info(f"[DEBUG] Batch {debug_steps}: Loss 计算完成 (耗时 {loss_elapsed:.4f}s)")
                    debug_steps += 1
            
            handle = interval_timer.start('backward')
            self.scaler.scale(total_loss).backward()
            interval_timer.stop('backward', handle)
            
            handle = interval_timer.start('step')
            self.scaler.step(self.optimizer)
            self.scaler.update()
            interval_timer.stop('step', handle)
            
            # 记录学习率
            lr_values.append(self.optimizer.param_groups[0]['lr'])

            total_loss_epoch += total_loss.item()
            
            handle = interval_timer.start('metrics')
            
            # 更新进度条
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                distill=f"{loss_dict.get('distill', 0):.4f}",
                edge=f"{loss_dict.get('edge', 0):.4f}",
                mask=f"{loss_dict.get('mask', 0):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                gpu_mem=f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}G"
            )
            
            batch_times.append(time.time() - t_data_start)
            
            interval_timer.stop('metrics', handle)
            
            # === 定期内存清理：每100个batch清理一次，避免内存碎片累积 ===
            if (batch_idx + 1) % 100 == 0:
                # 清理CUDA缓存（每100个batch清理一次，减少碎片）
                torch.cuda.empty_cache()
            
            # 开始下一个batch的数据加载计时
            t_data_start = time.time()
        
        # 关闭进度条
        progress_bar.close()
        
        # 打印平均耗时
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_data_load_time = np.mean(data_load_times) if data_load_times else 0
        logger.info(f"Epoch {epoch+1} training finished. Avg batch time: {avg_batch_time:.3f}s, Avg data load time: {avg_data_load_time:.3f}s")
        
        # 打印各阶段平均耗时
        durations = interval_timer.durations()
        duration_str = ", ".join([f"{k}: {v/num_batches*1000:.2f}ms" for k, v in durations.items()])
        logger.info(f"Avg timings per batch: {duration_str}")
        
        # 计算平均损失（必须在打印之前计算）
        avg_loss = total_loss_epoch / num_batches if num_batches > 0 else 0.0
        avg_distill_loss = distill_loss_epoch / num_batches if num_batches > 0 else 0.0
        avg_edge_loss = edge_loss_epoch / num_batches if (num_batches > 0 and self._is_edge_task_active(epoch)) else 0.0
        avg_mask_loss = mask_loss_epoch / num_batches if (num_batches > 0 and self._is_mask_task_active(epoch)) else 0.0
        
        # 打印详细的 Loss 信息
        logger.info(f"Epoch {epoch+1} Loss Summary:")
        logger.info(f"  Total Loss: {avg_loss:.6f}")
        logger.info(f"  Distill Loss: {avg_distill_loss:.6f}")
        if self._is_edge_task_active(epoch):
            logger.info(f"  Edge Loss: {avg_edge_loss:.6f}")
        if self._is_mask_task_active(epoch):
            logger.info(f"  Mask Loss: {avg_mask_loss:.6f}")
        else:
            logger.info(f"  Mask Loss: N/A (task starts at epoch {self.mask_task_start_epoch})")
        
        # 打印数据加载诊断统计
        recent_stats = load_diagnostics.get_recent_stats(last_n=100)
        logger.info(
            f"Last 100 batches data load stats: "
            f"Slowest: {recent_stats['slowest']:.3f}s, "
            f"Fastest: {recent_stats['fastest']:.3f}s, "
            f"Avg: {recent_stats['avg']:.3f}s, "
            f"Num Slow: {recent_stats['num_slow']}"
        )

        # === Epoch 结束清理 ===
        del load_diagnostics
        del interval_timer
        # 强制清理 Python 垃圾
        import gc
        gc.collect()
        # 清理 CUDA 缓存 (谨慎使用，因为会造成 sync 开销，建议每几个 epoch 清理一次)
        # 这里只在 epoch 结束时清理，影响不大
        torch.cuda.empty_cache()
        
        # 返回元组格式，与调用处期望的4个值匹配
        return (avg_loss, avg_distill_loss, avg_edge_loss, avg_mask_loss)
    
    def _handle_corrupt_npz(self, err: CorruptNPZError, loader: DataLoader, stage: str):
        """处理损坏的NPZ文件"""
        logger.warning(f"❌ [{stage}] Corrupted NPZ file detected: {err.file_path}")
        dataset = loader.dataset
        # 检查数据集是否有 'mark_corrupt' 方法
        if hasattr(dataset, 'mark_corrupt') and callable(getattr(dataset, 'mark_corrupt')):
            dataset.mark_corrupt(err.file_path)
            logger.info(f"Marked {err.file_path} as corrupt and will skip it in the future.")
        else:
            logger.warning(f"Dataset of type {type(dataset).__name__} does not support marking files as corrupt.")

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        compute_metrics: bool = True,
        desc: str = "Evaluating",
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        self.backbone.eval()
        self.edge_adapter.eval()
        self.edge_head.eval()
        self.feature_adapter.eval()
        
        # ✅ 学生的 Prompt Encoder 和 Mask Decoder 也设为评估模式
        if self.enable_mask_loss:
            if self.sam_prompt_encoder is not None:
                self.sam_prompt_encoder.eval()
            if self.sam_mask_decoder is not None:
                self.sam_mask_decoder.eval()

        edge_task_active = self._is_edge_task_active()
        mask_task_active = self._is_mask_task_active()

        # 【商业级优化】使用GPU tensor累加，避免每个batch都调用.item()
        totals = {
            "total": torch.tensor(0.0, device=self.device),
            "feat": torch.tensor(0.0, device=self.device),
            "edge": torch.tensor(0.0, device=self.device),
            "mask": torch.tensor(0.0, device=self.device),
        }
        batch_count = 0

        metrics_acc: dict[str, float] | None = None
        total_samples = 0
        mask_sample_count = 0
        if compute_metrics:
            metrics_acc = {
                "mse": 0.0,
                "mae": 0.0,
                "cosine_sim": 0.0,
                "edge_loss": 0.0,
                "mask_loss": 0.0,
                "mask_bce": 0.0,
                "mask_dice": 0.0,
                "mask_iou": 0.0,
            }

        # 安全地获取 loader 长度用于 tqdm（IterableDataset 没有 __len__）
        try:
            loader_len = len(loader)
        except (TypeError, NotImplementedError):
            loader_len = None
        pbar = tqdm(total=loader_len, desc=desc)
        data_iter = iter(loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"跳过一个批次，数据加载出错: {e}")
                if pbar.total:
                    pbar.total -= 1
                    pbar.refresh()
                continue

            # 修复 CUDA Graph 问题：在每个 batch 前标记步骤开始，防止 tensor 被覆盖
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            images_u8 = batch["image"]
            if images_u8.device != self.device:
                images_u8 = images_u8.to(self.device, non_blocking=True)
            # 【商业级优化】使用模型内的归一化层，避免循环内的Kernel Launch
            images = self.image_normalize(images_u8).contiguous(memory_format=torch.channels_last)
            teacher_features = batch["teacher_features"].to(self.device)
            edge_256x256 = batch["edge_256x256"].to(self.device)
            image_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [*batch["image_id"]]
            box_prompts_xyxy = batch.get("box_prompts_xyxy")
            box_prompts_masks = batch.get("box_prompts_masks_orig")
            box_prompts_count = batch.get("box_prompts_count")
            # 获取每个样本的原图尺寸
            image_shapes = batch.get("image_shape")  # 每个样本的原图尺寸
            if image_shapes is not None and not isinstance(image_shapes, list):
                # 如果是 tensor 或其他类型，转换为列表
                if isinstance(image_shapes, torch.Tensor):
                    image_shapes = [image_shapes[i].cpu().numpy() if image_shapes[i] is not None else None for i in range(len(image_shapes))]
                else:
                    image_shapes = [image_shapes] if image_shapes is not None else None
            fg_map = batch.get("fg_map")
            bg_map = batch.get("bg_map")
            edge_map = batch.get("edge_map")
            if isinstance(fg_map, torch.Tensor):
                fg_map = fg_map.to(self.device)
            if isinstance(bg_map, torch.Tensor):
                bg_map = bg_map.to(self.device)
            if isinstance(edge_map, torch.Tensor):
                if edge_map.numel() == 0:
                    edge_map = None
                else:
                    edge_map = edge_map.to(self.device)

            features = self.backbone(images)
            s4_features = features[0]
            s16_features = features[2]
            feat_loss, aligned_features = self._compute_distill_loss(
                s16_features,
                teacher_features,
                image_ids,
                return_aligned=True,
                fg_map=fg_map,
                bg_map=bg_map,
                edge_map=edge_map,
            )

            edge_loss_val = torch.zeros((), device=self.device)
            edge_loss_dict: dict = {}
            if edge_task_active:
                aligned_s4 = self.edge_adapter(s4_features)
                edge_logits = self.edge_head(aligned_s4)
                edge_loss_dict = self.edge_loss(edge_logits, edge_256x256, return_components=True)
                edge_loss_val = edge_loss_dict["total_loss"]

            logits = None
            mask_targets = None
            mask_loss_dict = None
            mask_loss_val = torch.zeros((), device=self.device)
            if mask_task_active and box_prompts_xyxy is not None:
                logits, mask_targets, valid_mask_vec = self._forward_mask_head(
                    aligned_features,
                    box_prompts_xyxy,
                    box_prompts_masks,
                    box_prompts_count,
                    image_shapes=image_shapes,  # 传递原图尺寸
                )
                if logits is not None and mask_targets is not None:
                    # 使用 valid_mask_vec 过滤有效样本计算 loss
                    valid_bool = valid_mask_vec > 0.5
                    if valid_bool.any():
                        valid_logits = logits[valid_bool]
                        valid_targets = mask_targets[valid_bool]
                        mask_loss_dict = self.mask_loss(valid_logits, valid_targets, return_components=True)
                        mask_loss_val = mask_loss_dict["total_loss"]
                    # 如果全是假框，mask_loss_val 保持为 0

            # 【商业级优化】在GPU上累加loss tensor，避免每个batch都调用.item()
            feat_loss_detached = feat_loss.detach()
            edge_loss_detached = edge_loss_val.detach() if edge_task_active else torch.tensor(0.0, device=self.device)
            mask_loss_detached = mask_loss_val.detach() if (self.enable_mask_loss and mask_task_active) else torch.tensor(0.0, device=self.device)

            # 累加tensor
            totals["feat"] = totals["feat"] + feat_loss_detached
            totals["edge"] = totals["edge"] + edge_loss_detached
            if self.enable_mask_loss:
                totals["mask"] = totals["mask"] + mask_loss_detached

            # 计算总loss（也在GPU上）
            total_loss_value = (
                self.config.get("feat_weight", 1.0) * feat_loss_detached
                + self.config.get("edge_weight", 1.0) * edge_loss_detached
            )
            if mask_task_active:
                total_loss_value = total_loss_value + self.mask_loss_weight * mask_loss_detached
            totals["total"] = totals["total"] + total_loss_value

            if compute_metrics:
                batch_size = images.size(0)
                # aligned_features 已通过 feature_adapter 对齐到正确尺寸，无需再次插值
                aligned_for_metrics = aligned_features

                # 优先使用 torchmetrics（在设备上累积，减少同步）
                if self.eval_metrics is not None:
                    self.eval_metrics.update(aligned_for_metrics, teacher_features)
                elif metrics_acc is not None:
                    # 【商业级优化】在GPU上累加tensor，避免.item()导致的CPU同步阻塞
                    # 移除flatten操作，避免OOM（直接计算空间维度的相似度）
                    mse = F.mse_loss(aligned_for_metrics, teacher_features, reduction="mean")
                    mae = F.l1_loss(aligned_for_metrics, teacher_features, reduction="mean")
                    # 使用空间维度计算余弦相似度，避免flatten导致OOM
                    # 对每个空间位置计算相似度，然后平均（避免创建巨大的flatten tensor）
                    cos_sim = F.cosine_similarity(
                        aligned_for_metrics.flatten(-2),  # 只flatten空间维度，保留batch和channel
                        teacher_features.flatten(-2),
                        dim=-1
                    ).mean()

                    # 在GPU上累加，最后只同步一次
                    metrics_acc["mse"] = metrics_acc.get("mse", torch.tensor(0.0, device=self.device)) + mse.detach() * batch_size
                    metrics_acc["mae"] = metrics_acc.get("mae", torch.tensor(0.0, device=self.device)) + mae.detach() * batch_size
                    metrics_acc["cosine_sim"] = metrics_acc.get("cosine_sim", torch.tensor(0.0, device=self.device)) + cos_sim.detach() * batch_size
                    total_samples += batch_size
                else:
                    # 初始化手动指标字典（使用GPU tensor）
                    metrics_acc = {
                        "mse": torch.tensor(0.0, device=self.device),
                        "mae": torch.tensor(0.0, device=self.device),
                        "cosine_sim": torch.tensor(0.0, device=self.device),
                        "edge_loss": 0.0,  # edge_loss保持float（因为可能在不同位置计算）
                        "mask_loss": 0.0,
                        "mask_bce": 0.0,
                        "mask_dice": 0.0,
                        "mask_iou": 0.0,
                    }
                    total_samples = 0

                # edge_loss 始终手动累积（因为它不是简单的特征对齐指标）
                if edge_task_active and metrics_acc is not None:
                    metrics_acc["edge_loss"] += edge_loss_val.item() * batch_size
                    if not hasattr(self, '_eval_total_samples'):
                        self._eval_total_samples = 0
                    self._eval_total_samples += batch_size

                if mask_task_active and mask_loss_dict is not None and logits is not None and mask_targets is not None:
                    current_count = logits.size(0)
                    if self.mask_metrics is not None:
                        # 使用 torchmetrics 计算 mask IOU
                        mask_probs = torch.sigmoid(logits.detach())
                        mask_binary = (mask_probs > 0.5).float()
                        self.mask_metrics.update(mask_binary, mask_targets)
                    elif metrics_acc is not None:
                        # 回退到手动计算
                        # 【商业级优化】在GPU上累加tensor，避免.item()导致的CPU同步阻塞
                        mask_sample_count += current_count
                        if "mask_loss" not in metrics_acc or not isinstance(metrics_acc["mask_loss"], torch.Tensor):
                            metrics_acc["mask_loss"] = torch.tensor(0.0, device=self.device)
                            metrics_acc["mask_bce"] = torch.tensor(0.0, device=self.device)
                            metrics_acc["mask_dice"] = torch.tensor(0.0, device=self.device)
                            metrics_acc["mask_iou"] = torch.tensor(0.0, device=self.device)
                        
                        metrics_acc["mask_loss"] = metrics_acc["mask_loss"] + mask_loss_val.detach() * current_count
                        metrics_acc["mask_bce"] = metrics_acc["mask_bce"] + mask_loss_dict["bce_loss"].detach() * current_count
                        metrics_acc["mask_dice"] = metrics_acc["mask_dice"] + mask_loss_dict["dice_loss"].detach() * current_count

                        mask_probs = torch.sigmoid(logits.detach())
                        mask_binary = (mask_probs > 0.5).float()
                        targets_binary = (mask_targets > 0.5).float()
                        intersection = (mask_binary * targets_binary).sum(dim=(1, 2, 3))
                        union = (mask_binary + targets_binary - mask_binary * targets_binary).sum(dim=(1, 2, 3)).clamp(min=1.0)
                        # 在GPU上累加IOU，最后统一转换
                        metrics_acc["mask_iou"] = metrics_acc["mask_iou"] + (intersection / union).sum()

            batch_count += 1
            pbar.update(1)

        pbar.close()

        denom = max(batch_count, 1)
        # 【商业级优化】最后只同步一次，将GPU tensor转换为float
        totals_avg = {k: (v / denom).item() if isinstance(v, torch.Tensor) else v / denom for k, v in totals.items()}
        if not self.enable_mask_loss:
            totals_avg["mask"] = 0.0

        metrics_out: dict[str, float] | None = None
        if compute_metrics:
            # 优先使用 torchmetrics（在设备上累积，最后同步一次）
            if self.eval_metrics is not None:
                eval_computed = self.eval_metrics.compute()
                metrics_out = {
                    "mse": float(eval_computed["mse"].item()),
                    "mae": float(eval_computed["mae"].item()),
                    "cosine_sim": float(eval_computed["cosine_sim"].item()),
                    "edge_loss": metrics_acc["edge_loss"] / max(getattr(self, '_eval_total_samples', total_samples), 1) if metrics_acc is not None else 0.0,
                }
                self.eval_metrics.reset()

                if mask_task_active and self.mask_metrics is not None:
                    mask_iou = self.mask_metrics.compute()
                    metrics_out.update({
                        "mask_iou": float(mask_iou.item()),
                    })
                    self.mask_metrics.reset()
                    # mask_loss/bce/dice 仍从 metrics_acc 读取（如果存在）
                    if metrics_acc is not None and mask_sample_count > 0:
                        metrics_out.update({
                            "mask_loss": metrics_acc["mask_loss"] / max(mask_sample_count, 1),
                            "mask_bce": metrics_acc["mask_bce"] / max(mask_sample_count, 1),
                            "mask_dice": metrics_acc["mask_dice"] / max(mask_sample_count, 1),
                        })
                    else:
                        metrics_out.update({
                            "mask_loss": 0.0,
                            "mask_bce": 0.0,
                            "mask_dice": 0.0,
                        })
                elif mask_task_active and self.mask_loss is not None:
                    if metrics_acc is not None and mask_sample_count > 0:
                        metrics_out.update({
                            "mask_loss": metrics_acc["mask_loss"] / max(mask_sample_count, 1),
                            "mask_bce": metrics_acc["mask_bce"] / max(mask_sample_count, 1),
                            "mask_dice": metrics_acc["mask_dice"] / max(mask_sample_count, 1),
                            "mask_iou": metrics_acc["mask_iou"] / max(mask_sample_count, 1),
                        })
                    else:
                        metrics_out.update({
                            "mask_loss": 0.0,
                            "mask_bce": 0.0,
                            "mask_dice": 0.0,
                            "mask_iou": 0.0,
                        })
                else:
                    metrics_out.update({
                        "mask_loss": 0.0,
                        "mask_bce": 0.0,
                        "mask_dice": 0.0,
                        "mask_iou": 0.0,
                    })
            elif metrics_acc is not None:
                # 【商业级优化】最后只同步一次，将GPU tensor转换为float
                # 避免每个batch都调用.item()导致CPU同步阻塞
                metrics_out = {
                    "mse": (metrics_acc["mse"] / max(total_samples, 1)).item(),
                    "mae": (metrics_acc["mae"] / max(total_samples, 1)).item(),
                    "cosine_sim": (metrics_acc["cosine_sim"] / max(total_samples, 1)).item(),
                    "edge_loss": metrics_acc["edge_loss"] / max(total_samples, 1),
                }
                if mask_task_active and self.mask_loss is not None:
                    if mask_sample_count > 0:
                        # 【商业级优化】最后只同步一次，将GPU tensor转换为float
                        metrics_out.update({
                            "mask_loss": (metrics_acc["mask_loss"] / max(mask_sample_count, 1)).item() if isinstance(metrics_acc["mask_loss"], torch.Tensor) else metrics_acc["mask_loss"] / max(mask_sample_count, 1),
                            "mask_bce": (metrics_acc["mask_bce"] / max(mask_sample_count, 1)).item() if isinstance(metrics_acc["mask_bce"], torch.Tensor) else metrics_acc["mask_bce"] / max(mask_sample_count, 1),
                            "mask_dice": (metrics_acc["mask_dice"] / max(mask_sample_count, 1)).item() if isinstance(metrics_acc["mask_dice"], torch.Tensor) else metrics_acc["mask_dice"] / max(mask_sample_count, 1),
                            "mask_iou": (metrics_acc["mask_iou"] / max(mask_sample_count, 1)).item() if isinstance(metrics_acc["mask_iou"], torch.Tensor) else metrics_acc["mask_iou"] / max(mask_sample_count, 1),
                        })
                    else:
                        metrics_out.update({
                            "mask_loss": 0.0,
                            "mask_bce": 0.0,
                            "mask_dice": 0.0,
                            "mask_iou": 0.0,
                        })
                else:
                    metrics_out.update({
                        "mask_loss": 0.0,
                        "mask_bce": 0.0,
                        "mask_dice": 0.0,
                        "mask_iou": 0.0,
                    })

        return totals_avg, metrics_out


def main():
    parser = argparse.ArgumentParser(description="Distill Single Test (MSE/FGD/FSDlike)")
    
    # 环境配置参数
    parser.add_argument("--env", type=str, default="ssh", choices=["ssh", "local"], 
                       help="环境类型：ssh使用SSH服务器路径，local使用本地路径")
    
    # 数据格式与数据集路径
    parser.add_argument("--data-format", type=str, default="tar_shard", choices=["npz_dir", "tar_shard", "binary"],
                       help="数据格式：tar_shard（默认，从tar shard读取，减少I/O开销）、npz_dir（从目录读取NPZ和图像）或binary（高性能定长二进制格式）")
    parser.add_argument("--features-dir", type=str, default=None, 
                       help="当 data-format=tar_shard 时为 shard 目录；当 data-format=npz_dir 时为 *_features.npz 目录")
    parser.add_argument("--images-dir", type=str, default=None, 
                       help="当 data-format=npz_dir 时为图像目录（1024x1024），tar_shard 模式不需要")
    parser.add_argument("--binary-data-path", type=str, default=None,
                       help="当 data-format=binary 时为二进制数据集根目录（包含 config.json 和所有 .bin 文件）")
    parser.add_argument("--verify-npz", action="store_true",
                       help="npz_dir 模式：启动前扫描NPZ文件并跳过损坏样本")
    parser.add_argument("--assume-clean", action="store_true",
                       help="npz_dir 模式：与 --verify-npz 搭配使用，使用已有缓存避免重复扫描")
    parser.add_argument("--bad-npz-cache", type=str, default=None,
                       help="npz_dir 模式：坏NPZ缓存文件路径（默认保存在features_dir/_bad_npz_cache.txt）")
    parser.add_argument("--gt-json-dir", type=str, default=None, help="SA风格GT目录，文件名为 sa_{image_id}.json")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "fgd", "fsdlike"]) 
    parser.add_argument("--backbone", type=str, default="rtdetrv2") 
    parser.add_argument("--student-backbone-type", type=str, default="PResNet",
                        choices=["PResNet", "HGNetv2", "CSPResNet", "TimmModel"],
                        help="选择 RT-DETR v2 学生骨干")
    parser.add_argument("--depth", type=int, default=50,
                        help="PResNet 深度（18、34、50、101）")
    parser.add_argument("--variant", type=str, default="L",
                        help="HGNetv2/CSPResNet/Timm 模型标识，例如 L/X/H、l/m/s 或 resnet50")
    parser.add_argument("--freeze-norm", action="store_true",
                        help="冻结骨干网络中的 BatchNorm 统计量")
    parser.add_argument("--student-pretrained", type=str, default=None,
                        help="学生骨干 ImageNet 预训练权重路径或 URL")
    parser.add_argument("--student-resume", type=str, default=None,
                        help="蒸馏阶段保存的学生骨干 checkpoint 路径")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size (默认16，SSH环境自动优化为32，RTX 3090建议 16-32)")
    parser.add_argument("--use-train-1200", action="store_true", help="使用固定的1200张训练数据（train_1200目录）")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8,
                       help="DataLoader的worker数量（默认8，SSH环境已优化，设置为0表示主进程加载）")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                       help="DataLoader的prefetch_factor（默认4，SSH环境已优化），需在num-workers>0时生效")
    parser.add_argument("--no-persistent-workers", action="store_true",
                       help="禁用persistent_workers（默认启用）")

    # FGD 超参 (官方默认值)
    parser.add_argument("--fgd-alpha-fg", type=float, default=0.001, help="前景权重")
    parser.add_argument("--fgd-beta-bg", type=float, default=0.0005, help="背景权重（前景的一半）")
    parser.add_argument("--fgd-alpha-edge", type=float, default=0.002, help="边缘权重（前景的两倍）")
    parser.add_argument("--fgd-gamma-mask", type=float, default=0.0)
    parser.add_argument("--fgd-lambda-rela", type=float, default=0.0)
    parser.add_argument("--fgd-temperature", type=float, default=1.0)

    # FSDlike 超参
    parser.add_argument("--fsd-weight-fg", type=float, default=1.0)
    parser.add_argument("--fsd-weight-bg", type=float, default=0.2)
    parser.add_argument("--fsd-temperature", type=float, default=1.0)
    parser.add_argument("--fsd-gaussian-from-mask", action="store_true", help="在掩码范围内生成并融合高斯权重")
    
    # 【新增】边缘增强参数
    parser.add_argument("--enable-edge-boost", action="store_true", help="启用边缘增强（FGD会单独计算边缘loss）")
    parser.add_argument("--fsd-gaussian-mix", type=str, default="max", choices=["max", "blend"])
    parser.add_argument("--fsd-gaussian-blend-lambda", type=float, default=0.5)
    
    # 【新增】渐进式边缘掩码参数
    parser.add_argument("--enable-edge-mask-progressive", action="store_true", 
                       help="启用渐进式边缘掩码（前N个epoch全局学习，之后只在膨胀边缘区域内计算损失）")
    parser.add_argument("--edge-mask-start-epoch", type=int, default=5,
                       help="从第几个epoch开始启用边缘掩码（默认5）")
    parser.add_argument("--edge-mask-kernel-size", type=int, default=3,
                       help="边缘膨胀核大小（默认3，即膨胀1像素）")
    
    # 【新增】正类重加权参数
    parser.add_argument("--use-pos-weight", action="store_true",
                       help="启用正类重加权（自动计算负正比作为pos_weight，解决类别不平衡）")
    parser.add_argument("--enable-mask-loss", action="store_true",
                       help="启用 SAM2 Prompt/Mask Decoder 蒸馏分支")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0,
                       help="SAM2 掩码损失在总损失中的权重")
    parser.add_argument("--mask-bce-weight", type=float, default=2.0,
                       help="SAM2 掩码分支 BCE 项权重")
    parser.add_argument("--mask-dice-weight", type=float, default=1.0,
                       help="SAM2 掩码分支 Dice 项权重")
    parser.add_argument("--mask-head-unfreeze-epoch", type=int, default=100,
                       help="SAM Prompt/Mask Decoder 在该 epoch 之后解冻训练（默认100，前期冻结）")
    parser.add_argument("--edge-task-start-epoch", type=int, default=5,
                       help="边缘蒸馏任务开始参与训练的epoch（默认5，意味着第5个epoch之后启用）")
    parser.add_argument("--mask-task-start-epoch", type=int, default=10,
                       help="掩码辨析蒸馏任务开始参与训练的epoch（默认10，意味着第10个epoch之后启用）")
    parser.add_argument("--enable-detailed-mask-logging", action="store_true",
                       help="启用详细的辨析头诊断日志（默认关闭）")
    
    # === 【新增】禁用编译开关 ===
    parser.add_argument("--no-compile", action="store_true",
                       help="禁用 torch.compile 加速，大幅减少启动等待时间（调试/测试推荐开启）")
    
    # === 【新增】手动指定数据总量 ===
    parser.add_argument("--total-images", type=int, default=109960, 
                       help="手动指定数据集总图像数（用于显示进度条ETA），默认109960")
    
    # 恢复训练与日志控制
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的checkpoint文件或输出目录路径")
    parser.add_argument("--strict-resume", action="store_true",
                       help="恢复训练时使用 strict=True 加载模型权重")
    parser.add_argument("--log-interval", type=int, default=50,
                       help="每隔多少个batch刷新一次耗时统计与进度显示")
    parser.add_argument("--teacher-channels", type=int, default=256,
                       help="教师特征通道数（默认256）")

    # CUDA设备选择
    parser.add_argument("--cuda-device", type=int, default=4,
                       help="指定使用的CUDA设备编号（默认4）")
    # 运行标签：用于区分同类实验的输出目录
    parser.add_argument("--run-tag", type=str, default="",
                       help="运行标签（例如实验名），将追加到输出目录后缀，用于隔离不同实验")
    
    # ========== 已禁用评估控制参数 ==========
    # parser.add_argument("--enable-eval", action="store_true",
    #                    help="启用每个 epoch 后的评估（默认关闭）")

    args = parser.parse_args()
    
    # ===== 根据环境设置优化的默认参数（针对52核CPU + 机械硬盘服务器） =====
    if args.env == "ssh":
        # 【强制修正】针对 HDD + RAM Cache 的最佳甜点位
        if args.batch_size == 16:
            args.batch_size = 32
            logger.info("[ENV] SSH环境：自动设置 batch-size=32")
        
        # 即使是 52 核 CPU，在 HDD 上开 8 个 worker 也是自杀 (磁头争抢)
        # 改为 4 个，配合 RAM Cache 策略，IO 效率最高
        force_workers = 4
        args.num_workers = force_workers
        logger.info(f"[ENV] SSH环境：强制设置 num-workers={force_workers} (HDD 最佳实践)")

        # 降低 prefetch，加快冷启动
        # 因为数据在 RAM 里，读起来很快，不需要预取太多
        force_prefetch = 2
        args.prefetch_factor = force_prefetch
        logger.info(f"[ENV] SSH环境：强制设置 prefetch-factor={force_prefetch} (加速启动)")
    
    # ===== Acceleration settings: enable cudnn benchmark and TF32 for faster training =====
    import torch as _torch
    _torch.backends.cudnn.benchmark = True
    # Allow TF32 on CUDA matmul (A40 supports TF32; improves throughput with minimal accuracy impact for MSE training)
    if hasattr(_torch.backends, 'cuda') and hasattr(_torch.backends.cuda, 'matmul'):
        _torch.backends.cuda.matmul.allow_tf32 = True
    # Prefer high precision mode for float32 matmul kernels where supported (PyTorch 2.x)
    try:
        _torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # 配置基础日志输出（控制台）
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 设置CUDA设备（在所有CUDA操作之前）
    if torch.cuda.is_available():
        if args.cuda_device < torch.cuda.device_count():
            torch.cuda.set_device(args.cuda_device)
            logger.info("[CUDA] 使用设备: cuda:%s (%s)", args.cuda_device, torch.cuda.get_device_name(args.cuda_device))
        else:
            logger.warning("指定的CUDA设备 %s 不存在，可用设备数量: %s", args.cuda_device, torch.cuda.device_count())
            logger.warning("使用默认设备: cuda:0")
            torch.cuda.set_device(0)
            args.cuda_device = 0
    else:
        logger.warning("CUDA不可用，使用CPU")
        args.cuda_device = None
    
    # 获取环境配置
    env_config = get_env_config(args.env)
    
    # 使用用户指定的路径或环境默认路径
    # 如果指定使用train_1200，则使用固定1200张数据
    if args.use_train_1200:
        features_dir = args.features_dir if args.features_dir else env_config["train_1200_features_dir"]
        images_dir = args.images_dir if args.images_dir else env_config["train_1200_images_dir"]
        logger.info("使用固定1200张训练数据（train_1200目录）")
    else:
        features_dir = args.features_dir if args.features_dir else env_config["features_dir"]
        images_dir = args.images_dir if args.images_dir else env_config["images_dir"]
    
    # 已禁用测试集相关配置
    # test_features_dir = env_config["test_features_dir"]
    # test_images_dir = env_config["test_images_dir"]
    output_base_dir = Path(env_config["output_base_dir"])
    
    # 检查必需路径
    if features_dir is None:
        raise ValueError("features_dir must be specified (use --features-dir or set env=ssh)")
    
    logger.info("[ENV] Environment: %s", args.env)
    logger.info("[ENV] Features dir (tar shards): %s", features_dir)
    logger.info("[ENV] Output base dir: %s", output_base_dir)

    config = {
        "loss_type": args.loss_type,
        "backbone": args.backbone,
        "student_backbone_type": args.student_backbone_type,
        "student_backbone_depth": args.depth,
        "student_backbone_variant": args.variant,
        "student_backbone_freeze_norm": args.freeze_norm,
        "student_pretrained": args.student_pretrained,
        "student_resume": args.student_resume,
        "learning_rate": args.lr,
        "feat_weight": args.feat_weight,
        "edge_weight": args.edge_weight,
        "teacher_channels": args.teacher_channels,
        "log_interval": args.log_interval,
        # FGD
        "fgd_alpha_fg": args.fgd_alpha_fg,
        "fgd_beta_bg": args.fgd_beta_bg,
        "fgd_alpha_edge": args.fgd_alpha_edge,
        "fgd_gamma_mask": args.fgd_gamma_mask,
        "fgd_lambda_rela": args.fgd_lambda_rela,
        "fgd_temperature": args.fgd_temperature,
        # FSD
        "fsd_weight_fg": args.fsd_weight_fg,
        "fsd_weight_bg": args.fsd_weight_bg,
        "fsd_temperature": args.fsd_temperature,
        "fsd_gaussian_from_mask": args.fsd_gaussian_from_mask,
        "fsd_gaussian_mix": args.fsd_gaussian_mix,
        "fsd_gaussian_blend_lambda": args.fsd_gaussian_blend_lambda,
        # 边缘增强与渐进式配置
        "enable_edge_boost": args.enable_edge_boost,
        "enable_edge_mask_progressive": args.enable_edge_mask_progressive,
        "edge_mask_start_epoch": args.edge_mask_start_epoch,
        "edge_mask_kernel_size": args.edge_mask_kernel_size,
        # 正类重加权
        "use_pos_weight": args.use_pos_weight,
        # 掩码蒸馏
        "enable_mask_loss": args.enable_mask_loss,
        "mask_loss_weight": args.mask_loss_weight,
        "mask_bce_weight": args.mask_bce_weight,
        "mask_dice_weight": args.mask_dice_weight,
        "mask_head_unfreeze_epoch": args.mask_head_unfreeze_epoch,
        "edge_task_start_epoch": args.edge_task_start_epoch,
        "mask_task_start_epoch": args.mask_task_start_epoch,
        "enable_detailed_mask_logging": args.enable_detailed_mask_logging,
        # === 【新增】禁用编译配置 ===
        "no_compile": args.no_compile,
    }
    
    resume_checkpoint: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume 路径不存在: {resume_path}")

        if resume_path.is_dir():
            output_dir = resume_path
            models_dir = output_dir / "models"
            if not models_dir.exists():
                raise FileNotFoundError(f"在 {models_dir} 下未找到模型目录，无法恢复训练")
            ckpts = list(models_dir.glob("epoch_*_model.pth"))
            if not ckpts:
                raise FileNotFoundError(f"{models_dir} 中没有可用的 checkpoint")

            def _extract_epoch(path: Path) -> int:
                parts = path.stem.split('_')
                for token in parts:
                    if token.isdigit():
                        return int(token)
                return -1

            resume_checkpoint = max(ckpts, key=_extract_epoch)
        else:
            resume_checkpoint = resume_path
            if resume_checkpoint.parent.name == "models":
                output_dir = resume_checkpoint.parent.parent
            else:
                output_dir = resume_checkpoint.parent

        logger.info("[RESUME] 将从 %s 恢复训练", resume_checkpoint)
    else:
        output_dir = create_output_directory(
            loss_type=args.loss_type,
            enable_edge_boost=args.enable_edge_boost,
            output_base_dir=output_base_dir,
            backbone=args.backbone,
            extra_suffix=args.run_tag,
        )

    # 确保输出子目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    log_dir = (output_dir / "models" / "logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("[OUTPUT] Output directory: %s", output_dir)
    logger.info("[OUTPUT] Log file: %s", log_file)
    logger.info("[OUTPUT] Models: %s", output_dir / "models")
    logger.info("[OUTPUT] Visualizations: %s", output_dir / "visualizations")

    # 根据 data_format 选择数据集类
    if args.data_format == "npz_dir":
        # ========== npz_dir 模式：从目录读取 NPZ 和图像 ==========
        logger.info("[INFO] 使用 npz_dir 数据格式")
        
        if features_dir is None:
            features_dir = env_config.get("features_dir", "/home/team/zouzhiyuan/dataset/sa1b/extracted")
        if images_dir is None:
            images_dir = env_config.get("images_dir", "/home/team/zouzhiyuan/dataset/sa1b")
        
        if not Path(features_dir).exists():
            logger.error(f"NPZ 目录不存在: {features_dir}")
            return
        if not Path(images_dir).exists():
            logger.error(f"图像目录不存在: {images_dir}")
            return
        
        train_dataset = NPZWithImageIdDataset(
            features_dir=features_dir,
            images_dir=images_dir,
            input_size=1024,
            verify_npz=args.verify_npz,
            bad_npz_cache=args.bad_npz_cache,
            verbose=True,
            assume_clean=args.assume_clean,
        )
        logger.info("[INFO] Training dataset (npz_dir):")
        logger.info("  features_dir=%s", features_dir)
        logger.info("  images_dir=%s", images_dir)
        logger.info("  total_samples=%s", len(train_dataset))
        
    elif args.data_format == "binary":
        # ========== binary 模式：从定长二进制文件读取（极速 IO） ==========
        logger.info("[INFO] 使用 binary 数据格式（高性能定长二进制存储）")
        
        # 二进制数据路径优先级：
        # 1. 用户指定的 --binary-data-path
        # 2. 环境配置中的 binary_data_dir（如果存在）
        # 3. 默认回退路径：/home/team/zouzhiyuan/dataset/sa1b_binary
        
        binary_data_path = args.binary_data_path
        if binary_data_path is None or not Path(binary_data_path).exists():
            # 尝试从环境配置获取
            binary_data_dir = env_config.get("binary_data_dir")
            if binary_data_dir and Path(binary_data_dir).exists():
                binary_data_path = binary_data_dir
                logger.info(f"[INFO] 使用环境配置的 binary_data_dir: {binary_data_path}")
            
            # 如果环境配置没有或不存在，尝试默认回退路径
            if binary_data_path is None or not Path(binary_data_path).exists():
                fallback_paths = [
                    "/home/team/zouzhiyuan/dataset/sa1b_binary",
                ]
                
                found_path = None
                for fallback_path in fallback_paths:
                    fallback_path_obj = Path(fallback_path)
                    if fallback_path_obj.exists():
                        # 检查是否有 config.json
                        config_file = fallback_path_obj / "config.json"
                        if config_file.exists():
                            found_path = fallback_path
                            logger.info(f"[INFO] 使用默认回退路径: {found_path}")
                            break
                
                if found_path:
                    binary_data_path = found_path
                else:
                    logger.error("无法找到有效的二进制数据集目录。请指定 --binary-data-path 或确保默认路径存在")
                    logger.error("默认路径: /home/team/zouzhiyuan/dataset/sa1b_binary")
                    if binary_data_path:
                        logger.error(f"指定的路径不存在: {binary_data_path}")
                    return
        
        # 最终检查：确保目录存在且包含 config.json
        binary_path = Path(binary_data_path)
        if not binary_path.exists():
            logger.error(f"二进制数据集目录不存在: {binary_data_path}")
            return
        
        config_file = binary_path / "config.json"
        if not config_file.exists():
            logger.error(f"二进制数据集配置文件不存在: {config_file}")
            logger.error("请先运行 convert_tar_to_bin.py 生成二进制数据集")
            return
        
        # 创建二进制数据集
        train_dataset = BinaryDistillDataset(
            data_root=str(binary_data_path),
            input_size=1024,
            verbose=True,
        )
        logger.info("[INFO] Training dataset (binary):")
        logger.info("  data_root=%s", binary_data_path)
        logger.info("  total_samples=%s", len(train_dataset))
        logger.info("  🚀 使用极速二进制 IO（O(1) Seek，Zero-Copy）")
        
    elif args.data_format == "tar_shard":
        # ========== tar_shard 模式：从 tar 文件流式读取 ==========
        logger.info("[INFO] 使用 tar_shard 数据格式")
        
        # tar_shard 模式的路径优先级：
        # 1. 用户指定的 --features-dir
        # 2. 环境配置中的 tar_shard 目录（如果存在）
        # 3. 默认回退路径：/home/team/zouzhiyuan/dataset/sa1b_tar_shards
        
        if features_dir is None or not Path(features_dir).exists():
            # 尝试从环境配置获取 tar_shard 目录
            tar_shard_dir = env_config.get("tar_shard_dir")
            if tar_shard_dir and Path(tar_shard_dir).exists():
                tar_files = list(Path(tar_shard_dir).glob("*.tar*"))
                if len(tar_files) > 0:
                    features_dir = tar_shard_dir
                    logger.info(f"[INFO] 使用环境配置的 tar_shard_dir: {features_dir} (找到 {len(tar_files)} 个 tar 文件)")
            
            # 如果环境配置没有或不存在，尝试默认回退路径
            if features_dir is None or not Path(features_dir).exists():
                fallback_paths = [
                    "/home/team/zouzhiyuan/dataset/sa1b_tar_shards",
                ]
                
                found_path = None
                for fallback_path in fallback_paths:
                    fallback_path_obj = Path(fallback_path)
                    if fallback_path_obj.exists():
                        # 检查是否有 tar 文件
                        tar_files = list(fallback_path_obj.glob("*.tar*"))
                        if len(tar_files) > 0:
                            found_path = fallback_path
                            logger.info(f"[INFO] 使用默认回退路径: {found_path} (找到 {len(tar_files)} 个 tar 文件)")
                            break
                
                if found_path:
                    features_dir = found_path
                else:
                    logger.error("无法找到有效的 tar_shard 目录。请指定 --features-dir 或确保默认路径存在")
                    logger.error("默认路径: /home/team/zouzhiyuan/dataset/sa1b_tar_shards")
                    if features_dir:
                        logger.error(f"指定的路径不存在: {features_dir}")
                    return
        
        # 最终检查：确保目录存在且包含 tar 文件
        features_path = Path(features_dir)
        if not features_path.exists():
            logger.error(f"tar_shard 目录不存在: {features_dir}")
            return
        
        tar_files = list(features_path.glob("*.tar*"))
        if len(tar_files) == 0:
            logger.error(f"tar_shard 目录中没有找到 tar 文件: {features_dir}")
            return
        
        logger.info(f"[INFO] 最终使用 tar_shard 目录: {features_dir} (找到 {len(tar_files)} 个 tar 文件)")
        
        # 使用流式 Dataset 优化机械硬盘 IO
        train_dataset = StreamingTarDataset(
            shard_dir=features_dir,
            input_size=1024,
            shuffle_buffer_size=0,  # 极速 IO 模式：直通，无 Buffer 等待
            verbose=True,
            use_ram_cache=True # <--- 确保这一项是 True (默认也是 True)
        )
        logger.info("[INFO] Training dataset (streaming tar_shard):")
        logger.info("  shard_dir=%s", features_dir)
        logger.info("  shuffle_buffer_size=%s", train_dataset.shuffle_buffer_size)
        # IterableDataset 没有 __len__，无法直接获取总数
    else:
        raise ValueError(f"不支持的数据格式: {args.data_format}")
        
    # ========== 已禁用测试集相关逻辑 ==========
    # test_dataset = None
    # if args.enable_eval:
    #     # 测试集：优先找 test shard，否则回退到 npz_dir（测试集1000张，I/O压力小）
    #     test_shard_dir = env_config.get("test_shard_dir", "/home/team/zouzhiyuan/dataset/sa1b_tar_shards_test")
    #     test_shard_exists = False
    #     if test_shard_dir and Path(test_shard_dir).exists():
    #         # 检查是否有 tar 文件
    #         test_shard_files = list(Path(test_shard_dir).glob("*.tar*"))
    #         if test_shard_files:
    #             test_shard_exists = True
    #     
    #     if test_shard_exists:
    #         logger.info("[INFO] Test 使用 tar_shard")
    #         test_dataset = TarShardNPZDataset(
    #             shard_dir=test_shard_dir,
    #             input_size=1024,
    #             verbose=True,
    #         )
    #         logger.info("[INFO] Test dataset (tar_shard):")
    #         logger.info("  test_shard_dir=%s", test_shard_dir)
    #         logger.info("  test_total_samples=%s", len(test_dataset))
    # else:
    #     logger.info("[INFO] 已禁用评估或未配置测试 shard，跳过测试集加载")
    
    # 数据加载器配置：支持自定义worker/prefetch/persistent
    persistent_workers = (not args.no_persistent_workers) and args.num_workers > 0
    pin_device = ""
    if torch.cuda.is_available():
        if args.cuda_device is not None and args.cuda_device < torch.cuda.device_count():
            pin_device = f"cuda:{args.cuda_device}"
        else:
            pin_device = f"cuda:{torch.cuda.current_device()}"
    # IterableDataset 必须设置 shuffle=False（随机性由 Dataset 内部的 shuffle buffer 提供）
    # ===== 强制优化 num_workers (针对机械硬盘 Tar 读取) =====
    # 【已禁用】对于 RAM 缓存策略，worker 数量已在 SSH 环境优化中强制设置为 4
    # 不再强制增加 workers，避免与 RAM 缓存优化冲突
    # 对于机械硬盘 Tar 读取，解压是 CPU 密集型，同时读取需要一定的并发掩盖 IO 延迟
    # 但过多的并发会导致磁头反复寻道（Thrashing）。
    # 经验值：8-12 workers (取决于 CPU 核数)，prefetch_factor 设为 2-4
    # optimal_workers = 8
    # if args.num_workers < optimal_workers:
    #     logger.warning(f"[IO OPTIMIZATION] Force increasing num_workers from {args.num_workers} to {optimal_workers} for tar performance.")
    #     args.num_workers = optimal_workers
    
    num_workers_for_hdd = args.num_workers if args.num_workers > 0 else 0
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset 不支持 DataLoader 的 shuffle
        num_workers=num_workers_for_hdd,
        pin_memory=True,
    )
    # ========== 已禁用测试集 DataLoader 配置 ==========
    # test_loader_kwargs = dict(
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    if pin_device:
        train_loader_kwargs["pin_memory_device"] = pin_device
        # test_loader_kwargs["pin_memory_device"] = pin_device
    if num_workers_for_hdd > 0:
        train_loader_kwargs["persistent_workers"] = True  # 流式读取建议开启
        # test_loader_kwargs["persistent_workers"] = persistent_workers
        # 【已禁用】对于 RAM 缓存策略，prefetch_factor 已在 SSH 环境优化中强制设置为 2
        # 不再强制增加 prefetch，避免与 RAM 缓存优化冲突
        # ✅ 强制优化 prefetch_factor
        # 每个 worker 预取 4 个 batch，保证 GPU 不饥饿
        # optimal_prefetch = 4
        # if args.prefetch_factor < optimal_prefetch:
        #     logger.warning(f"[IO OPTIMIZATION] Force increasing prefetch_factor from {args.prefetch_factor} to {optimal_prefetch}.")
        #     args.prefetch_factor = optimal_prefetch
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        # if args.prefetch_factor > 0:
        #     test_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        collate_fn=_custom_collate_fn,  # ✅ 使用自定义 collate_fn 处理变长张量
        worker_init_fn=_loader_worker_init if num_workers_for_hdd > 0 else None,
        **train_loader_kwargs,
    )
    val_loader = None  # 已取消验证集
    # ========== 已禁用测试集 DataLoader 创建 ==========
    # test_loader = None
    # if args.enable_eval and test_dataset is not None:
    #     test_loader = DataLoader(
    #         test_dataset,
    #         worker_init_fn=_loader_worker_init if args.num_workers > 0 else None,
    #         **test_loader_kwargs,
    #     )
    
    logger.info("")
    logger.info("[INFO] DataLoader Configuration:")
    logger.info("  Train: num_workers=%s, prefetch_factor=%s, persistent_workers=%s, pin_memory=%s",
                train_loader_kwargs.get('num_workers', 0),
                train_loader_kwargs.get('prefetch_factor', 0),
                train_loader_kwargs.get('persistent_workers', False),
                train_loader_kwargs.get('pin_memory', False))
    # ========== 已禁用测试集日志输出 ==========
    # if test_loader is not None:
    #     logger.info("  Test:  num_workers=%s, prefetch_factor=%s, persistent_workers=%s, pin_memory=%s",
    #                 test_loader_kwargs.get('num_workers', 0),
    #                 test_loader_kwargs.get('prefetch_factor', 0),
    #                 test_loader_kwargs.get('persistent_workers', False),
    #                 test_loader_kwargs.get('pin_memory', False))
    logger.info("")
    logger.info("[INFO] 数据集划分:")
    logger.info("  train=流式读取 (IterableDataset 无固定长度)")
    # ========== 已禁用测试集数据集划分日志 ==========
    # if args.enable_eval and test_dataset is not None:
    #     logger.info("  test=%s (固定1000张测试集)", len(test_dataset))

    # 创建设备对象（基于设置的CUDA设备）
    if torch.cuda.is_available() and args.cuda_device is not None:
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(args.cuda_device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    
    # 【内存优化】配置CUDA内存分配策略，避免训练后期OOM
    if device.type == "cuda":
        # 1. 设置内存分配器以减小碎片
        import os
        # 设置max_split_size_mb来减少内存碎片（128MB块大小，避免大块分配失败）
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        
        # 2. 清理CUDA缓存
        torch.cuda.empty_cache()
        
        # 3. 设置内存增长策略（如果支持）
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, device=device)
        except Exception:
            pass
        
        # 4. 打印内存信息
        total_mem = torch.cuda.get_device_properties(device.index).total_memory / (1024**3)
        logger.info(f"[CUDA] 使用设备: {device} (NVIDIA {torch.cuda.get_device_name(device.index)})")
        logger.info(f"[CUDA] 总显存: {total_mem:.2f} GB")
        logger.info(f"[CUDA] 内存优化: max_split_size_mb=128 (减少碎片)")
    
    runner = DistillSingleTester(config, device=device)
    
    start_epoch = 0  # 从 0 开始，显示时使用 epoch+1 以符合人类习惯（第1个epoch）
    if resume_checkpoint is not None:
        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("[RESUME] Loading checkpoint from %s", resume_checkpoint)
        logger.info("%s", "=" * 70)
        checkpoint = torch.load(resume_checkpoint, map_location=runner.device)

        strict_flag = bool(args.strict_resume)

        def load_component(key: str, module: nn.Module, friendly: str) -> None:
            if key not in checkpoint:
                logger.warning("WARNING: checkpoint缺少 %s, 跳过 %s 恢复", key, friendly)
                return
            state_dict = checkpoint[key]

            try:
                module.load_state_dict(state_dict, strict=strict_flag)
            except RuntimeError as exc:
                if strict_flag:
                    raise
                logger.warning("%s strict 加载失败，使用 strict=False 重试: %s", friendly, exc)
                module.load_state_dict(state_dict, strict=False)
            else:
                logger.info("✅ %s loaded", friendly)

        load_component('backbone', runner.backbone, 'Backbone')
        load_component('edge_head', runner.edge_head, 'EdgeHead')
        load_component('edge_adapter', runner.edge_adapter, 'EdgeAdapter')
        load_component('feature_adapter', runner.feature_adapter, 'FeatureAdapter')
        
        # ✅ 加载学生的 Prompt Encoder 和 Mask Decoder 权重（如果启用掩码损失）
        if runner.enable_mask_loss:
            # 获取配置（优先使用 checkpoint 中的 config，否则使用 runner.config）
            resume_config = checkpoint.get("config", runner.config)
            sam2_head_config = resume_config.get("sam2_head_config", runner.config.get("sam2_head_config", "sam2.1/sam2.1_hiera_b+.yaml"))
            sam2_head_ckpt = resume_config.get("sam2_head_ckpt", runner.config.get("sam2_head_ckpt", None))
            
            if 'sam_prompt_encoder' in checkpoint and runner.sam_prompt_encoder is not None:
                load_component('sam_prompt_encoder', runner.sam_prompt_encoder, 'SAM Prompt Encoder (Student)')
            elif runner.sam_prompt_encoder is not None:
                # 如果没有保存的权重，从教师初始化（兼容旧 checkpoint）
                logger.info("[RESUME] Checkpoint 中没有 sam_prompt_encoder，使用教师权重初始化（从 %s）", 
                           sam2_head_ckpt or "weights/sam2.1_hiera_base_plus.pt")
                try:
                    # 重新加载教师权重
                    pe, _, _ = runner.load_sam2_heads(
                        device=str(runner.device),
                        config_file=sam2_head_config,
                        ckpt_path=sam2_head_ckpt,
                        return_model=False,
                    )
                    pe_weights = pe.state_dict()
                    runner.sam_prompt_encoder.load_state_dict(pe_weights, strict=False)
                    logger.info("✅ SAM Prompt Encoder (Student) initialized from teacher weights")
                except Exception as exc:
                    logger.warning("[RESUME] 无法从教师初始化 sam_prompt_encoder: %s", exc)
            
            if 'sam_mask_decoder' in checkpoint and runner.sam_mask_decoder is not None:
                load_component('sam_mask_decoder', runner.sam_mask_decoder, 'SAM Mask Decoder (Student)')
            elif runner.sam_mask_decoder is not None:
                # 如果没有保存的权重，从教师初始化（兼容旧 checkpoint）
                logger.info("[RESUME] Checkpoint 中没有 sam_mask_decoder，使用教师权重初始化（从 %s）", 
                           sam2_head_ckpt or "weights/sam2.1_hiera_base_plus.pt")
                try:
                    # 重新加载教师权重
                    _, md, _ = runner.load_sam2_heads(
                        device=str(runner.device),
                        config_file=sam2_head_config,
                        ckpt_path=sam2_head_ckpt,
                        return_model=False,
                    )
                    md_weights = md.state_dict()
                    runner.sam_mask_decoder.load_state_dict(md_weights, strict=False)
                    logger.info("✅ SAM Mask Decoder (Student) initialized from teacher weights")
                except Exception as exc:
                    logger.warning("[RESUME] 无法从教师初始化 sam_mask_decoder: %s", exc)

        if 'optimizer' in checkpoint:
            runner.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("✅ Optimizer loaded")
        else:
            logger.warning("WARNING: checkpoint中没有optimizer状态，使用当前优化器默认值")

        if 'scaler' in checkpoint:
            runner.scaler.load_state_dict(checkpoint['scaler'])
            logger.info("✅ GradScaler loaded")
        else:
            logger.warning("WARNING: checkpoint中没有scaler状态，缩放因子将重新初始化")

        if 'epoch' in checkpoint:
            # checkpoint 中保存的是已完成的 epoch（从0开始计数）
            # 所以从 checkpoint['epoch'] + 1 开始训练
            start_epoch = int(checkpoint['epoch']) + 1
            logger.info("▶️  Training will resume from epoch %s (displayed as Epoch %s)", start_epoch, start_epoch + 1)

        logger.info("%s", "=" * 70)
        logger.info("")
    else:
        logger.info("[INFO] 从头开始训练")
    
    # 日志文件路径
    log_file = output_dir / "logs" / "results.log"
    # 确保 logs 目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 训练循环（每个epoch都保存模型，用于快速测试）
    save_interval = 1
    epoch_times: list[float] = []
    
    # 最佳模型跟踪（用于交付）- 仅基于训练loss
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_metric_name: str = "train_total"  # 固定使用训练loss

    # === 【新增】计算总 batch 数（如果提供了 total-images） ===
    total_batches_est = None
    if args.total_images is not None:
        # 这里的 args.batch_size 已经是优化后的值（如果是SSH环境，之前代码会自动改为32）
        total_batches_est = args.total_images // args.batch_size
        logger.info(f"[INFO] 手动设定进度条: 总图像 {args.total_images:,} / Batch {args.batch_size} ≈ {total_batches_est:,} Batches")

    # 注意：epoch 内部从 0 开始计数，但显示时用 epoch+1（更符合人类习惯）
    if start_epoch >= args.epochs:
        logger.warning("start_epoch (%s) 已大于等于总训练轮数 (%s)，跳过训练阶段", start_epoch, args.epochs)
    else:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            # === 【修改】传入 total_batches_est ===
            tr_tot, tr_feat, tr_edge, tr_mask = runner.train_epoch(train_loader, epoch, external_total_batches=total_batches_est)
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            
            # 将结果追加写入日志文件（包含epoch时间）
            with log_file.open("a", encoding="utf-8") as lf:
                if runner.enable_mask_loss and runner.mask_loss is not None:
                    lf.write(
                        f"epoch={epoch}\ttrain_total={tr_tot:.6f}\ttrain_feat={tr_feat:.6f}\ttrain_edge={tr_edge:.6f}\t"
                        f"train_mask={tr_mask:.6f}\tepoch_time={epoch_time:.2f}s\n"
                    )
                else:
                    lf.write(f"epoch={epoch}\ttrain_total={tr_tot:.6f}\ttrain_feat={tr_feat:.6f}\ttrain_edge={tr_edge:.6f}\tepoch_time={epoch_time:.2f}s\n")
            
            # 每个epoch都保存模型（save_interval=1）
            should_save = (epoch % save_interval == 0) or (epoch == args.epochs)
            model_save_path: Path | None = None
            clean_state: dict[str, torch.Tensor] | None = None
            
            if should_save:
                # === 保存前清理缓存，避免checkpoint保存时OOM ===
                torch.cuda.empty_cache()
                
                # === 1) 完整断点：用于 resume、包含优化器等所有状态 ===
                model_save_path = output_dir / "models" / f"epoch_{epoch}_model.pth"
                save_dict = {
                    "backbone": runner.backbone.state_dict(),
                    "edge_adapter": runner.edge_adapter.state_dict(),
                    "edge_head": runner.edge_head.state_dict(),
                    "feature_adapter": runner.feature_adapter.state_dict(),
                    "optimizer": runner.optimizer.state_dict(),
                    "scaler": runner.scaler.state_dict(),
                    "epoch": epoch,
                    "config": config,
                }
                
                # ✅ 保存学生的 Prompt Encoder 和 Mask Decoder 权重（如果启用掩码损失）
                if runner.enable_mask_loss:
                    if runner.sam_prompt_encoder is not None:
                        save_dict["sam_prompt_encoder"] = runner.sam_prompt_encoder.state_dict()
                    if runner.sam_mask_decoder is not None:
                        save_dict["sam_mask_decoder"] = runner.sam_mask_decoder.state_dict()
                
                torch.save(save_dict, model_save_path)
                logger.info("[SAVE] Saved full checkpoint (epoch %s) to %s", epoch, model_save_path)
                if runner.enable_mask_loss:
                    if runner.sam_prompt_encoder is not None:
                        logger.info("[SAVE] Saved sam_prompt_encoder (student) weights")
                    if runner.sam_mask_decoder is not None:
                        logger.info("[SAVE] Saved sam_mask_decoder (student) weights")

                # === 2) 清洗后的 Backbone：用于下游 MMDet 直接加载 ===
                clean_backbone_path = output_dir / "models" / f"epoch_{epoch}_backbone_mmdet.pth"
                # 修复前缀顺序：先匹配长前缀，再匹配短前缀
                clean_state = runner.export_clean_backbone_state(prefixes_to_strip=("backbone.backbone.", "backbone."))
                clean_checkpoint = {
                    "meta": {
                        "epoch": epoch,
                        "exported_from": "train_distill_single_test",
                        "backbone": config.get("backbone", "rtdetrv2"),
                        "model_size": config.get("model_size", "s"),
                    },
                    "state_dict": clean_state,
                    # ✅ 保存 feature_adapter，供下游选择使用
                    "adapter_state_dict": runner.feature_adapter.state_dict(),
                }
                torch.save(clean_checkpoint, clean_backbone_path)
                logger.info("✅ [EXPORT] Saved MMDet-compatible backbone to %s", clean_backbone_path)
            
            # 使用训练loss作为最佳模型指标（每个epoch都判断）
            current_metric_value = tr_tot
            
            # 判断并保存最佳模型
            is_best = False
            if best_metric_value is None:
                # 第一个epoch，直接作为最佳
                is_best = True
                best_metric_value = current_metric_value
                best_epoch = epoch
            elif current_metric_value < best_metric_value:
                # 发现更好的模型（loss越小越好）
                is_best = True
                best_metric_value = current_metric_value
                best_epoch = epoch
            
            if is_best:
                logger.info("")
                logger.info("🏆 [BEST] New best model at epoch %s (train_total=%.6f, previous=%.6f)", 
                           epoch, current_metric_value, 
                           best_metric_value if best_epoch != epoch else float('inf'))
                
                # === 保存最佳完整checkpoint ===
                best_model_path = output_dir / "models" / "best_model.pth"
                # 如果当前epoch已保存，直接复制；否则需要重新保存
                if should_save and model_save_path is not None:
                    import shutil
                    shutil.copy2(model_save_path, best_model_path)
                    logger.info("✅ [BEST] Saved best full checkpoint to %s", best_model_path)
                else:
                    # 如果当前epoch未保存，需要重新保存完整checkpoint
                    best_save_dict = {
                        "backbone": runner.backbone.state_dict(),
                        "edge_adapter": runner.edge_adapter.state_dict(),
                        "edge_head": runner.edge_head.state_dict(),
                        "feature_adapter": runner.feature_adapter.state_dict(),
                        "optimizer": runner.optimizer.state_dict(),
                        "scaler": runner.scaler.state_dict(),
                        "epoch": epoch,
                        "config": config,
                    }
                    if runner.enable_mask_loss:
                        if runner.sam_prompt_encoder is not None:
                            best_save_dict["sam_prompt_encoder"] = runner.sam_prompt_encoder.state_dict()
                        if runner.sam_mask_decoder is not None:
                            best_save_dict["sam_mask_decoder"] = runner.sam_mask_decoder.state_dict()
                    torch.save(best_save_dict, best_model_path)
                    logger.info("✅ [BEST] Saved best full checkpoint to %s", best_model_path)
                
                # === 保存最佳清洗后的Backbone ===
                best_backbone_path = output_dir / "models" / "best_backbone_mmdet.pth"
                # 如果clean_state已存在（当前epoch已保存），直接使用；否则重新生成
                if clean_state is not None:
                    best_clean_state = clean_state
                else:
                    # 重新生成clean_state
                    best_clean_state = runner.export_clean_backbone_state(prefixes_to_strip=("backbone.backbone.", "backbone."))
                
                best_clean_checkpoint = {
                    "meta": {
                        "epoch": epoch,
                        "exported_from": "train_distill_single_test",
                        "backbone": config.get("backbone", "rtdetrv2"),
                        "model_size": config.get("model_size", "s"),
                        "best_metric": "train_total",
                        "best_metric_value": float(current_metric_value),
                    },
                    "state_dict": best_clean_state,
                    "adapter_state_dict": runner.feature_adapter.state_dict(),
                }
                torch.save(best_clean_checkpoint, best_backbone_path)
                logger.info("✅ [BEST] Saved best MMDet-compatible backbone to %s", best_backbone_path)
                logger.info("")
    
    # 记录训练总时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    logger.info("")
    logger.info("%s", "=" * 50)
    logger.info("[TRAINING TIME] 训练完成")
    logger.info("  总训练时间: %.2f 小时 (%.2f 分钟)", total_training_time / 3600, total_training_time / 60)
    logger.info("  平均每epoch: %.2f 分钟", avg_epoch_time / 60 if avg_epoch_time else 0.0)
    logger.info("%s", "=" * 50)
    logger.info("")
    
    logger.info("")
    logger.info("%s", "=" * 50)
    logger.info("[TRAINING COMPLETE] 训练完成，模型已保存")
    if best_epoch is not None and best_metric_value is not None:
        logger.info("")
        logger.info("🏆 [BEST MODEL] 最佳模型信息:")
        logger.info("  Epoch: %s", best_epoch)
        logger.info("  指标: %s = %.6f", best_metric_name, best_metric_value)
        logger.info("  完整checkpoint: %s", output_dir / "models" / "best_model.pth")
        logger.info("  MMDet兼容backbone: %s", output_dir / "models" / "best_backbone_mmdet.pth")
        logger.info("  (包含feature_adapter，供下游选择使用)")
    else:
        logger.info("  ⚠️  未找到最佳模型（可能所有epoch都未完成评估）")
    logger.info("%s", "=" * 50)


def visualize_results(runner, dataset, args, config, output_dir: Path):
    """Visualize edge maps and feature maps after training"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # 使用传入的输出目录下的visualizations子目录
    vis_output_dir = output_dir / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[VIS] Output directory: %s", vis_output_dir)
    
    # 设置为评估模式
    runner.backbone.eval()
    runner.edge_head.eval()
    
    # Select first 10 images for visualization
    num_vis = min(10, len(dataset))
    indices = list(range(num_vis))
    
    logger.info("[VIS] Visualizing %s images...", num_vis)
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(runner.device)
            teacher_features = sample['teacher_features'].unsqueeze(0).to(runner.device)
            edge_gt = sample['edge_256x256']
            image_id = sample['image_id']
            
            # 前向传播
            features = runner.backbone(image)
            s4_features = features[0]  # S4: 256×256，用于边缘头
            s16_features = features[2]  # S16: 64×64，用于特征对齐
            
            # Align to teacher feature scale (使用S16用于特征可视化)
            aligned_features = runner.feature_adapter(s16_features)
            if aligned_features.shape[-2:] != teacher_features.shape[-2:]:
                aligned_features = F.interpolate(
                    aligned_features,
                    size=teacher_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            
            # Generate edge map (使用EdgeAdapter对齐S4特征，然后输入边缘头)
            aligned_s4 = runner.edge_adapter(s4_features)  # [B, 128, 256, 256] -> [B, 256, 256, 256]
            edge_logits = runner.edge_head(aligned_s4)  # [B, 256, 256, 256] -> [B, 1, 256, 256]
            edge_pred = torch.sigmoid(edge_logits[0, 0]).cpu().numpy()  # (256, 256)
            
            # P4 feature visualization (aligned features)
            p4_feat = aligned_features[0].cpu().numpy()  # (256, 64, 64)
            p4_mean = p4_feat.mean(axis=0)  # (64, 64)
            p4_energy = np.sqrt((p4_feat ** 2).mean(axis=0))  # (64, 64)
            
            # Teacher feature visualization
            teacher_feat = teacher_features[0].cpu().numpy()  # (256, 64, 64)
            teacher_mean = teacher_feat.mean(axis=0)  # (64, 64)
            
            # Original image
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)  # (1024, 1024, 3)
            img_np = np.clip(img_np, 0, 1)
            
            # Create figure
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)
            
            # Row 1: original, edge GT, edge pred, overlay, error
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_np)
            ax0.set_title(f"Original Image (ID: {image_id})", fontsize=10)
            ax0.axis('off')
            
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(edge_gt.numpy(), cmap='gray', vmin=0, vmax=1)
            ax1.set_title("Edge GT (256x256)", fontsize=10)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(edge_pred, cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f"Edge Prediction\n(mean={edge_pred.mean():.3f})", fontsize=10)
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 3])
            # Overlay on original
            img_resized = torch.nn.functional.interpolate(
                torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0),
                size=(256, 256), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0).numpy()
            ax3.imshow(img_resized)
            ax3.contour(edge_pred, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
            ax3.set_title("Edge Overlay (thr=0.5)", fontsize=10)
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[0, 4])
            edge_diff = np.abs(edge_pred - edge_gt.numpy())
            im4 = ax4.imshow(edge_diff, cmap='hot', vmin=0, vmax=1)
            ax4.set_title(f"Edge Error\n(MAE={edge_diff.mean():.3f})", fontsize=10)
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            ax4.axis('off')
            
            # Row 2: P4 mean, P4 energy, teacher mean, diff, channel grid
            ax5 = fig.add_subplot(gs[1, 0])
            im5 = ax5.imshow(p4_mean, cmap='viridis')
            ax5.set_title(f"P4 Mean (64x64)\nmean={p4_mean.mean():.3f}", fontsize=10)
            plt.colorbar(im5, ax=ax5, fraction=0.046)
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 1])
            im6 = ax6.imshow(p4_energy, cmap='hot')
            ax6.set_title(f"P4 Energy\nmax={p4_energy.max():.3f}", fontsize=10)
            plt.colorbar(im6, ax=ax6, fraction=0.046)
            ax6.axis('off')
            
            ax7 = fig.add_subplot(gs[1, 2])
            im7 = ax7.imshow(teacher_mean, cmap='viridis')
            ax7.set_title(f"Teacher Mean\nmean={teacher_mean.mean():.3f}", fontsize=10)
            plt.colorbar(im7, ax=ax7, fraction=0.046)
            ax7.axis('off')
            
            ax8 = fig.add_subplot(gs[1, 3])
            feat_diff = np.abs(p4_mean - teacher_mean)
            im8 = ax8.imshow(feat_diff, cmap='hot')
            ax8.set_title(f"Feature Diff\nMAE={feat_diff.mean():.3f}", fontsize=10)
            plt.colorbar(im8, ax=ax8, fraction=0.046)
            ax8.axis('off')
            
            ax9 = fig.add_subplot(gs[1, 4])
            # First 16 channels grid
            n_show = min(16, p4_feat.shape[0])
            grid_size = 4
            channel_grid = np.zeros((grid_size * 16, grid_size * 16))
            for i in range(n_show):
                row, col = i // grid_size, i % grid_size
                ch_data = p4_feat[i]
                ch_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(ch_data).unsqueeze(0).unsqueeze(0),
                    size=(16, 16), mode='bilinear', align_corners=False
                )[0, 0].numpy()
                channel_grid[row*16:(row+1)*16, col*16:(col+1)*16] = ch_resized
            im9 = ax9.imshow(channel_grid, cmap='gray')
            ax9.set_title(f"First {n_show} Channels", fontsize=10)
            ax9.axis('off')
            
            # 保存
            save_path = vis_output_dir / f"{image_id}_visualization.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  [VIS] %s/%s: %s", idx + 1, num_vis, save_path.name)
    
    logger.info("")
    logger.info("Visualization completed. Saved to: %s", vis_output_dir)
    logger.info("%s", "=" * 50)
    logger.info("")


if __name__ == "__main__":
    main()



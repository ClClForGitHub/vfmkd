#!/usr/bin/env python3
"""
单独测试 MSE/FGD/FSDlike 蒸馏损失的最小训练脚本。
参考 VFMKD/tools/warmup_training_v1.py 的数据/日志风格，仅替换特征对齐损失。
"""

import os
import sys
import warnings
import argparse
import time
from pathlib import Path
from typing import List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

# 屏蔽Windows/Numpy等运行时告警，保持训练输出干净
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# 添加项目根目录到路径（从tools/core/向上两级到VFMKD/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.edge_head import UniversalEdgeHead
from vfmkd.distillation.losses.feature_loss import FeatureLoss
from vfmkd.distillation.losses.fgd_loss import FGDLoss
from vfmkd.distillation.losses.fsd_loss import FSDLikeLoss
from vfmkd.distillation.losses.edge_loss import EdgeDistillationLoss
from vfmkd.distillation.adapters import SimpleAdapter
from vfmkd.distillation.gt_adapter import build_batch_bboxes_from_ids, build_fg_bg_from_ids


# ==================== 环境配置 ====================
def get_env_config(env_type: str = "ssh"):
    """
    根据环境类型返回数据集和输出路径配置
    
    Args:
        env_type: "ssh" 或 "local"
    
    Returns:
        dict: 包含features_dir, images_dir, output_base_dir的配置
    """
    if env_type == "ssh":
        return {
            "features_dir": "/home/team/zouzhiyuan/dataset/sa1b/extracted",
            "images_dir": "/home/team/zouzhiyuan/dataset/sa1b",
            "output_base_dir": "/home/team/zouzhiyuan/vfmkd/outputs",
        }
    elif env_type == "local":
        # TODO: 实现local环境的路径配置
        return {
            "features_dir": None,  # 需要用户指定
            "images_dir": None,   # 需要用户指定
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
    
    print(f"\n{'='*60}")
    print(f"Checkpoint结构分析: {checkpoint_path}")
    print(f"{'='*60}")
    
    # 基本信息
    if 'epoch' in checkpoint:
        print(f"训练epoch: {checkpoint['epoch']}")
    if 'config' in checkpoint:
        print(f"配置信息: 已包含")
    
    # 权重模块
    print(f"\n保存的权重模块:")
    for key in ['backbone', 'edge_head', 'feature_adapter', 'optimizer']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            num_params = sum(p.numel() for p in state_dict.values()) if isinstance(state_dict, dict) else 0
            num_keys = len(state_dict) if isinstance(state_dict, dict) else 0
            print(f"  ✓ {key}: {num_keys} 个参数组, 总参数量: {num_params:,}")
            
            # 如果是feature_adapter，列出内部的适配器
            if key == 'feature_adapter':
                adapter_keys = [k for k in state_dict.keys() if k.startswith('adapters.')]
                if adapter_keys:
                    print(f"    内部适配器: {len(adapter_keys)} 个")
                    for adapter_key in adapter_keys[:5]:  # 只显示前5个
                        print(f"      - {adapter_key}")
                    if len(adapter_keys) > 5:
                        print(f"      ... 还有 {len(adapter_keys) - 5} 个适配器")
        else:
            print(f"  ✗ {key}: 未找到")
    
    print(f"{'='*60}\n")
    return checkpoint


def create_output_directory(loss_type: str, enable_edge_boost: bool, output_base_dir: Path, 
                           backbone: str = "yolov8", extra_suffix: str = "") -> Path:
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
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    return output_dir


class NPZWithImageIdDataset(Dataset):
    def __init__(self, features_dir: str, images_dir: str | None = None, max_images: int | None = None, input_size: int = 1024):
        self.features_dir = Path(features_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.input_size = input_size
        npz_files = list(self.features_dir.glob("*_features.npz"))
        if max_images:
            npz_files = npz_files[:max_images]
        if self.images_dir is None:
            raise ValueError("images_dir must be provided and point to the real images directory (no random placeholders allowed).")

        self.valid_files: List[Path] = []
        self.missing_images: List[str] = []
        for f in npz_files:
            try:
                data = np.load(f)
                # 兼容新旧NPZ格式：P4_S16 (新) 或 IMAGE_EMB_S16 (旧)
                has_teacher = "P4_S16" in data or "IMAGE_EMB_S16" in data
                has_edge = "edge_256x256" in data
                if not (has_teacher and has_edge):
                    continue
                image_id = f.stem.replace("_features", "")
                if self._find_image_path(image_id) is not None:
                    self.valid_files.append(f)
                else:
                    self.missing_images.append(image_id)
            except Exception:
                continue

        if len(self.valid_files) == 0:
            raise ValueError("No valid samples found with matching real images. Please check images_dir and NPZ naming.")
        if len(self.missing_images) > 0:
            print(f"[WARN] {len(self.missing_images)} samples skipped due to missing images. Examples: {self.missing_images[:5]}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        npz_file = self.valid_files[idx]
        image_id = npz_file.stem.replace("_features", "")
        data = np.load(npz_file)
        
        # 兼容新旧NPZ格式
        if "P4_S16" in data:
            teacher_features = torch.from_numpy(data["P4_S16"]).float().squeeze(0)  # (256,64,64)
        else:
            teacher_features = torch.from_numpy(data["IMAGE_EMB_S16"]).float().squeeze(0)  # (256,64,64)
        
        edge_256x256 = torch.from_numpy(data["edge_256x256"]).float()
        # 加载真实图片（严格要求）
        image_tensor = self._load_real_image(image_id)
        if image_tensor is None:
            raise FileNotFoundError(f"Image file for image_id={image_id} not found under {self.images_dir}")
        return {
            "image": image_tensor,
            "teacher_features": teacher_features,
            "edge_256x256": edge_256x256,
            "image_id": image_id,
        }

    def _find_image_path(self, image_id: str):
        cand = [
            f"sa_{image_id}.jpg", f"{image_id}.jpg",
            f"sa_{image_id}.png", f"{image_id}.png",
            f"{image_id}.jpeg", f"sa_{image_id}.jpeg",
        ]
        for name in cand:
            p = self.images_dir / name
            if p.exists():
                return p
        return None

    def _load_real_image(self, image_id: str) -> torch.Tensor | None:
        img_path = self._find_image_path(image_id)
        if img_path is None:
            return None
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img


class DistillSingleTester:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # backbone
        self.backbone = self._create_backbone().to(self.device).train()

        # heads / adapters
        self.edge_head = UniversalEdgeHead(
            core_channels=config.get("core_channels", 256),
            output_channels=1,
            head_type=config.get("head_type", "simple"),
            init_p=0.05,
        ).to(self.device)
        self.feature_adapter = SimpleAdapter().to(self.device)
        
        # 边缘损失函数（支持渐进式边缘掩码和正类重加权）
        self.edge_loss = EdgeDistillationLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            enable_edge_mask=False,  # 初始关闭，由enable_edge_mask_progressive控制
            edge_mask_kernel_size=config.get("edge_mask_kernel_size", 3),
            use_pos_weight=config.get("use_pos_weight", False),
        ).to(self.device)
        
        # 渐进式边缘掩码配置
        self.enable_edge_mask_progressive = config.get("enable_edge_mask_progressive", False)
        self.edge_mask_start_epoch = config.get("edge_mask_start_epoch", 5)
        self.current_epoch = 0

        # distill loss
        loss_type = config.get("loss_type", "mse")
        if loss_type == "mse":
            self.distill_loss = FeatureLoss({"loss_type": "mse", "alpha": config.get("mse_weight", 1.0)}).to(self.device)
        elif loss_type == "fgd":
            self.distill_loss = FGDLoss(
                alpha_fg=config.get("fgd_alpha_fg", 0.001),  # 官方默认
                beta_bg=config.get("fgd_beta_bg", 0.0005),  # 官方默认（前景的一半）
                alpha_edge=config.get("fgd_alpha_edge", 0.002),  # 前景的两倍
                gamma_mask=config.get("fgd_gamma_mask", 0.0),
                lambda_rela=config.get("fgd_lambda_rela", 0.0),
                temperature=config.get("fgd_temperature", 1.0),
            ).to(self.device)
        elif loss_type == "fsdlike":
            self.distill_loss = FSDLikeLoss(
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

        all_params = list(self.backbone.parameters()) + list(self.edge_head.parameters()) + list(self.feature_adapter.parameters())
        self.optimizer = optim.Adam(all_params, lr=config.get("learning_rate", 1e-3))

    def _create_backbone(self) -> nn.Module:
        backbone_type = self.config.get("backbone", "yolov8")
        if backbone_type == "yolov8":
            return YOLOv8Backbone({
                "model_size": "s",
                "pretrained": self.config.get("pretrained", False),
                "freeze_backbone": self.config.get("freeze_backbone", False),
                "freeze_at": self.config.get("freeze_at", -1),
                "checkpoint_path": self.config.get("checkpoint_path", None),
            })
        elif backbone_type == "repvit":
            return RepViTBackbone({
                "arch": "m1",
                "img_size": 1024,
                "fuse": False,
                "freeze": self.config.get("freeze_backbone", False),
                "load_from": self.config.get("checkpoint_path", None),
            })
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

    def _compute_distill_loss(self, p3_features, teacher_features, image_ids: List[str]):
        # 将学生p3与教师特征对齐到教师尺度
        aligned_features = self.feature_adapter(p3_features, teacher_features)
        loss_type = self.config.get("loss_type", "mse")
        if loss_type == "mse":
            return self.distill_loss(aligned_features, teacher_features)

        # 需要GT: 优先从NPZ加载预计算的权重，否则从JSON实时计算
        Hf, Wf = aligned_features.shape[-2], aligned_features.shape[-1]
        npz_dir = self.config.get("npz_feature_dir", None)
        fg_map, bg_map = None, None
        
        # 尝试从NPZ加载预计算的权重
        if npz_dir:
            try:
                from vfmkd.distillation.gt_adapter import load_weights_from_npz
                fg_map, bg_map = load_weights_from_npz(image_ids, npz_dir, (Hf, Wf))
                fg_map = fg_map.to(aligned_features.device)
                bg_map = bg_map.to(aligned_features.device)
            except Exception as e:
                print(f"⚠️  Failed to load weights from NPZ: {e}, falling back to JSON")
                fg_map, bg_map = None, None
        
        # 如果NPZ中没有权重，从JSON实时计算
        if fg_map is None or bg_map is None:
            gt_dir = self.config.get("gt_json_dir", None)
            if not gt_dir:
                # 无GT时，退化为MSE以确保可运行
                return F.mse_loss(aligned_features, teacher_features)
            
            # 基于JSON分割生成前景/背景权重
            fg_map, bg_map = build_fg_bg_from_ids(image_ids, gt_dir, (Hf, Wf))  # (B,1,Hf,Wf)
            fg_map = fg_map.to(aligned_features.device)
            bg_map = bg_map.to(aligned_features.device)
        
        # 【新增】加载边缘图（用于FGD的边缘loss）
        edge_map = None
        enable_edge_boost = self.config.get("enable_edge_boost", False)
        if enable_edge_boost and loss_type == "fgd":
            npz_dir = self.config.get("npz_feature_dir", None)
            
            if npz_dir:
                from vfmkd.distillation.gt_adapter import load_edge_maps_from_npz
                
                # 加载边缘图
                edge_map = load_edge_maps_from_npz(
                    image_ids, 
                    npz_dir, 
                    (Hf, Wf),
                    use_real_edge=False,
                    real_edge_maps=None
                )
                edge_map = edge_map.to(aligned_features.device)

        if loss_type == "fgd":
            return self.distill_loss(aligned_features, teacher_features, fg_map=fg_map, bg_map=bg_map, edge_map=edge_map)
        elif loss_type == "fsdlike":
            return self.distill_loss(aligned_features, teacher_features, fg_map=fg_map, bg_map=bg_map)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.backbone.train()
        self.edge_head.train()
        self.feature_adapter.train()
        tot, tot_feat, tot_edge = 0.0, 0.0, 0.0
        
        # 更新当前epoch
        self.current_epoch = epoch
        
        # 渐进式启用边缘掩码（从第N个epoch开始）
        if self.enable_edge_mask_progressive and epoch >= self.edge_mask_start_epoch:
            if not self.edge_loss.enable_edge_mask:
                print("\n" + "="*60)
                print(f"[Epoch {epoch}] Enable edge-region mask")
                print("   Edge loss computed only within dilated edge band (3x3 kernel)")
                print("   Focus on precise boundary alignment; ignore interior textures")
                print("="*60 + "\n")
                self.edge_loss.enable_edge_mask = True
        elif not self.enable_edge_mask_progressive:
            # 如果没有启用渐进模式，则保持初始状态
            pass
        
        # Timing statistics
        timing_stats = {
            'data_load': 0.0,
            'backbone': 0.0,
            'distill_loss': 0.0,
            'edge_head': 0.0,
            'edge_loss': 0.0,
            'backward': 0.0,
            'optimizer': 0.0,
            'total_batch': 0.0
        }
        batch_count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        batch_start = time.time()
        
        for batch in pbar:
            t0 = time.time()
            images = batch["image"].to(self.device)
            teacher_features = batch["teacher_features"].to(self.device)
            edge_256x256 = batch["edge_256x256"].to(self.device)
            image_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [*batch["image_id"]]
            timing_stats['data_load'] += time.time() - t0

            self.optimizer.zero_grad()
            
            t0 = time.time()
            features = self.backbone(images)
            # features[0]=s4(256x256), features[1]=s8(128x128), features[2]=s16(64x64), features[3]=s32(32x32)
            # S4用于边缘头（256×256），S16用于特征蒸馏（匹配教师特征64×64）
            s4_features = features[0]  # S4: 256×256分辨率，用于边缘头
            s16_features = features[2]  # S16: 64×64分辨率，用于特征蒸馏
            timing_stats['backbone'] += time.time() - t0

            t0 = time.time()
            feat_loss = self._compute_distill_loss(s16_features, teacher_features, image_ids)
            timing_stats['distill_loss'] += time.time() - t0

            t0 = time.time()
            # 使用S4特征（256×256）输入边缘头，输出直接是256×256，无需上采样
            edge_logits = self.edge_head(s4_features, backbone_name=self.config.get("backbone", "yolov8"))
            # edge_logits已经是256×256，不需要上采样
            timing_stats['edge_head'] += time.time() - t0
            
            t0 = time.time()
            edge_loss_dict = self.edge_loss(edge_logits, edge_256x256, return_components=True)
            edge_loss_val = edge_loss_dict["total_loss"]
            timing_stats['edge_loss'] += time.time() - t0
            
            # 记录边缘掩码覆盖率（如果启用了掩码）
            if 'edge_mask_ratio' in edge_loss_dict:
                if not hasattr(self, '_edge_mask_ratio_sum'):
                    self._edge_mask_ratio_sum = 0.0
                    self._edge_mask_ratio_count = 0
                self._edge_mask_ratio_sum += edge_loss_dict['edge_mask_ratio']
                self._edge_mask_ratio_count += 1
            
            # 记录正类权重（如果启用了pos_weight）
            if 'pos_weight' in edge_loss_dict:
                if not hasattr(self, '_pos_weight_sum'):
                    self._pos_weight_sum = 0.0
                    self._pos_weight_count = 0
                self._pos_weight_sum += edge_loss_dict['pos_weight']
                self._pos_weight_count += 1

            total_loss = self.config.get("feat_weight", 1.0) * feat_loss + self.config.get("edge_weight", 1.0) * edge_loss_val
            
            t0 = time.time()
            total_loss.backward()
            timing_stats['backward'] += time.time() - t0
            
            t0 = time.time()
            self.optimizer.step()
            timing_stats['optimizer'] += time.time() - t0

            timing_stats['total_batch'] += time.time() - batch_start
            batch_count += 1
            batch_start = time.time()

            tot += total_loss.item()
            tot_feat += feat_loss.item()
            tot_edge += edge_loss_val.item()
            pbar.set_postfix({"total": f"{total_loss.item():.4f}", "feat": f"{feat_loss.item():.4f}", "edge": f"{edge_loss_val.item():.4f}"})

        n = len(loader)
        
        # Print timing statistics
        if batch_count > 0:
            print(f"\n{'='*60}")
            print(f"Timing Statistics (Epoch {epoch}, {batch_count} batches):")
            print(f"{'='*60}")
            for key, total_time in timing_stats.items():
                avg_time = total_time / batch_count
                percentage = (total_time / timing_stats['total_batch'] * 100) if timing_stats['total_batch'] > 0 else 0
                print(f"  {key:15s}: {avg_time:7.3f}s/batch  ({total_time:7.2f}s total, {percentage:5.1f}%)")
            print(f"{'='*60}\n")
        
        # Print edge mask statistics
        if hasattr(self, '_edge_mask_ratio_sum') and self._edge_mask_ratio_count > 0:
            avg_mask_ratio = self._edge_mask_ratio_sum / self._edge_mask_ratio_count
            print(f"Edge mask coverage: {avg_mask_ratio*100:.2f}% (avg across {self._edge_mask_ratio_count} batches)")
            # 重置统计
            self._edge_mask_ratio_sum = 0.0
            self._edge_mask_ratio_count = 0
        
        # Print pos_weight statistics
        if hasattr(self, '_pos_weight_sum') and self._pos_weight_count > 0:
            avg_pos_weight = self._pos_weight_sum / self._pos_weight_count
            print(f"Pos_weight (neg/pos ratio): {avg_pos_weight:.2f}x (avg across {self._pos_weight_count} batches)")
            # 重置统计
            self._pos_weight_sum = 0.0
            self._pos_weight_count = 0
        
        print()
        
        return tot / n, tot_feat / n, tot_edge / n

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.backbone.eval()
        self.edge_head.eval()
        self.feature_adapter.eval()
        tot, tot_feat, tot_edge = 0.0, 0.0, 0.0
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(self.device)
            teacher_features = batch["teacher_features"].to(self.device)
            edge_256x256 = batch["edge_256x256"].to(self.device)
            image_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [*batch["image_id"]]

            features = self.backbone(images)
            # features[0]=s4(256x256), features[1]=s8(128x128), features[2]=s16(64x64), features[3]=s32(32x32)
            s4_features = features[0]  # S4: 256×256，用于边缘头
            s16_features = features[2]  # S16: 64×64，用于特征蒸馏
            feat_loss = self._compute_distill_loss(s16_features, teacher_features, image_ids)

            # 使用S4特征（256×256）输入边缘头，输出直接是256×256，无需上采样
            edge_logits = self.edge_head(s4_features, backbone_name=self.config.get("backbone", "yolov8"))
            # edge_logits已经是256×256，不需要上采样
            edge_loss_dict = self.edge_loss(edge_logits, edge_256x256, return_components=True)
            edge_loss_val = edge_loss_dict["total_loss"]

            total_loss = self.config.get("feat_weight", 1.0) * feat_loss + self.config.get("edge_weight", 1.0) * edge_loss_val
            tot += total_loss.item()
            tot_feat += feat_loss.item()
            tot_edge += edge_loss_val.item()
        n = len(loader)
        return tot / n, tot_feat / n, tot_edge / n
    
    @torch.no_grad()
    def validate_unified_metrics(self, loader: DataLoader):
        """使用统一指标评估：MSE、MAE、余弦相似度"""
        self.backbone.eval()
        self.edge_head.eval()
        self.feature_adapter.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_cos_sim = 0.0
        total_edge_iou = 0.0
        total_samples = 0
        
        for batch in tqdm(loader, desc="Unified Metrics"):
            images = batch["image"].to(self.device)
            teacher_features = batch["teacher_features"].to(self.device)
            edge_256x256 = batch["edge_256x256"].to(self.device)
            
            batch_size = images.size(0)
            
            # 前向传播
            features = self.backbone(images)
            s4_features = features[0]  # S4: 256×256，用于边缘头
            s16_features = features[2]  # S16: 64×64，用于特征蒸馏
            aligned_features = self.feature_adapter(s16_features, teacher_features)
            
            # 特征指标 - 使用mean reduction，然后乘以batch_size累加
            mse = F.mse_loss(aligned_features, teacher_features, reduction='mean').item()
            mae = F.l1_loss(aligned_features, teacher_features, reduction='mean').item()
            
            # 余弦相似度 - 对每个样本计算，然后求平均
            student_flat = aligned_features.flatten(1)
            teacher_flat = teacher_features.flatten(1)
            cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1).mean().item()
            
            total_mse += mse * batch_size
            total_mae += mae * batch_size
            total_cos_sim += cos_sim * batch_size
            
            # 边缘指标 - 使用S4特征（256×256）输入边缘头，输出直接是256×256
            edge_logits = self.edge_head(s4_features, backbone_name=self.config.get("backbone", "yolov8"))
            # edge_logits已经是256×256，不需要上采样
            
            # 使用与训练相同的edge loss计算
            edge_loss_dict = self.edge_loss(edge_logits, edge_256x256, return_components=True)
            edge_loss_val = edge_loss_dict["total_loss"].item()
            
            total_edge_iou += edge_loss_val * batch_size
            
            total_samples += batch_size
        
        return {
            'mse': total_mse / total_samples,
            'mae': total_mae / total_samples,
            'cosine_sim': total_cos_sim / total_samples,
            'edge_loss': total_edge_iou / total_samples  # 改名为edge_loss
        }


def main():
    parser = argparse.ArgumentParser(description="Distill Single Test (MSE/FGD/FSDlike)")
    
    # 环境配置参数
    parser.add_argument("--env", type=str, default="ssh", choices=["ssh", "local"], 
                       help="环境类型：ssh使用SSH服务器路径，local使用本地路径")
    
    # 数据集路径（如果指定则覆盖环境默认值）
    parser.add_argument("--features-dir", type=str, default=None, 
                       help="NPZ特征目录（默认从env配置获取）")
    parser.add_argument("--images-dir", type=str, default=None, 
                       help="图像目录（默认从env配置获取）")
    parser.add_argument("--gt-json-dir", type=str, default=None, help="SA风格GT目录，文件名为 sa_{image_id}.json")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "fgd", "fsdlike"]) 
    parser.add_argument("--backbone", type=str, default="yolov8", choices=["yolov8", "repvit"]) 
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="最多读取多少个NPZ，用于快速测试")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--no-val", action="store_true", help="不做验证，全部样本用于训练")

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
    
    # 【新增】恢复训练/仅测试参数
    parser.add_argument("--resume", type=str, default=None, 
                       help="加载checkpoint路径（设置epochs=0可仅测试）")

    args = parser.parse_args()
    
    # 获取环境配置
    env_config = get_env_config(args.env)
    
    # 使用用户指定的路径或环境默认路径
    features_dir = args.features_dir if args.features_dir else env_config["features_dir"]
    images_dir = args.images_dir if args.images_dir else env_config["images_dir"]
    output_base_dir = Path(env_config["output_base_dir"])
    
    # 检查必需路径
    if features_dir is None:
        raise ValueError("features_dir must be specified (use --features-dir or set env=ssh)")
    if images_dir is None:
        raise ValueError("images_dir must be specified (use --images-dir or set env=ssh)")
    
    print(f"[ENV] Environment: {args.env}")
    print(f"[ENV] Features dir: {features_dir}")
    print(f"[ENV] Images dir: {images_dir}")
    print(f"[ENV] Output base dir: {output_base_dir}")

    config = {
        "loss_type": args.loss_type,
        "backbone": args.backbone,
        "learning_rate": args.lr,
        "feat_weight": args.feat_weight,
        "edge_weight": args.edge_weight,
        "gt_json_dir": args.gt_json_dir,
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
        # 【新增】边缘增强
        "enable_edge_boost": args.enable_edge_boost,
        "npz_feature_dir": features_dir,  # 用于加载边缘图
        # 【新增】渐进式边缘掩码
        "enable_edge_mask_progressive": args.enable_edge_mask_progressive,
        "edge_mask_start_epoch": args.edge_mask_start_epoch,
        "edge_mask_kernel_size": args.edge_mask_kernel_size,
        # 【新增】正类重加权
        "use_pos_weight": args.use_pos_weight,
    }
    
    # 创建输出目录（在训练开始前创建，确保日志可以写入）
    output_dir = create_output_directory(
        loss_type=args.loss_type,
        enable_edge_boost=args.enable_edge_boost,
        output_base_dir=output_base_dir,
        backbone=args.backbone,
    )
    print(f"[OUTPUT] Output directory: {output_dir}")
    print(f"[OUTPUT] Logs: {output_dir / 'logs'}")
    print(f"[OUTPUT] Models: {output_dir / 'models'}")
    print(f"[OUTPUT] Visualizations: {output_dir / 'visualizations'}")

    # 数据集加载（不再限制300张，可以加载所有可用数据）
    dataset = NPZWithImageIdDataset(features_dir, images_dir, max_images=args.max_images, input_size=1024)
    print(f"[INFO] features_dir={features_dir}")
    print(f"[INFO] images_dir={images_dir}")
    print(f"[INFO] gt_json_dir={args.gt_json_dir}")
    print(f"[INFO] npz_total={len(dataset)}")
    if len(dataset) == 0:
        print("[ERROR] 未找到 *_features.npz，有效样本为0，请检查目录与文件命名。")
        return
    
    # 数据集划分策略：固定测试集用于对比不同训练方法
    # 训练集：270张 (90%), 测试集：30张 (10%)
    # 使用固定随机种子确保测试集一致
    torch.manual_seed(42)
    test_size = int(0.1 * len(dataset))  # 10% 作为固定测试集
    train_val_size = len(dataset) - test_size
    train_val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if args.no_val:
        # 无验证集模式：所有训练+验证数据用于训练，测试集保留
        train_dataset = train_val_dataset
        val_dataset = None
        print(f"[INFO] split: train={len(train_dataset)}, test={len(test_dataset)} (no-val mode)")
    else:
        # 有验证集模式：从train_val中再分出验证集
        train_size = int(0.9 * len(train_val_dataset))  # 90% of train_val for training
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[INFO] split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    runner = DistillSingleTester(config)
    
    # 加载checkpoint（如果指定）
    if args.resume:
        print(f"\n[RESUME] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=runner.device)
        runner.backbone.load_state_dict(checkpoint['backbone'])
        runner.edge_head.load_state_dict(checkpoint['edge_head'], strict=False)
        runner.feature_adapter.load_state_dict(checkpoint['feature_adapter'])
        if 'optimizer' in checkpoint:
            runner.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[RESUME] Checkpoint loaded successfully!")
        if args.epochs == 0:
            print(f"[RESUME] epochs=0, skipping training, going directly to test...")
    
    # 日志文件路径
    log_file = output_dir / "logs" / "results.log"
    
    # 训练循环（每20个epoch保存一次模型）
    save_interval = 20  # 每20个epoch保存一次
    
    for epoch in range(1, args.epochs + 1):
        tr_tot, tr_feat, tr_edge = runner.train_epoch(train_loader, epoch)
        if val_loader is not None:
            va_tot, va_feat, va_edge = runner.validate(val_loader)
            print(f"Epoch {epoch}: Train total={tr_tot:.4f} feat={tr_feat:.4f} edge={tr_edge:.4f} | Val total={va_tot:.4f} feat={va_feat:.4f} edge={va_edge:.4f}")
        else:
            va_tot = va_feat = va_edge = float('nan')
            print(f"Epoch {epoch}: Train total={tr_tot:.4f} feat={tr_feat:.4f} edge={tr_edge:.4f}")
        
        # 将结果追加写入日志文件
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"epoch={epoch}\ttrain_total={tr_tot:.6f}\ttrain_feat={tr_feat:.6f}\ttrain_edge={tr_edge:.6f}\tval_total={va_tot}\tval_feat={va_feat}\tval_edge={va_edge}\n")
        
        # 每20个epoch保存一次模型，以及最后一个epoch
        should_save = (epoch % save_interval == 0) or (epoch == args.epochs)
        if should_save:
            model_save_path = output_dir / "models" / f"epoch_{epoch}_model.pth"
            # 保存的模型权重包含：
            # 1. backbone: 骨干网络（YOLOv8/RepViT）的权重
            # 2. edge_head: 边缘预测头的权重
            # 3. feature_adapter: 特征对齐适配器（SimpleAdapter）的权重
            #    - SimpleAdapter内部使用ModuleDict动态管理适配器
            #    - feature_adapter.state_dict()会自动包含所有已创建的适配器权重
            #    - 例如：adapters.{student_channels}to{teacher_channels}.conv.weight/bias
            # 4. optimizer: 优化器状态（用于恢复训练）
            # 5. epoch: 当前epoch编号
            # 6. config: 训练配置字典
            torch.save({
                "backbone": runner.backbone.state_dict(),
                "edge_head": runner.edge_head.state_dict(),
                "feature_adapter": runner.feature_adapter.state_dict(),
                "optimizer": runner.optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            }, model_save_path)
            print(f"[SAVE] Saved model (epoch {epoch}) to {model_save_path}")
    
    # 训练完成后在测试集上评估
    print("\n" + "="*50)
    print("[TEST] Evaluating on test set...")
    test_tot, test_feat, test_edge = runner.validate(test_loader)
    print(f"Test Results: total={test_tot:.4f} feat={test_feat:.4f} edge={test_edge:.4f}")
    
    # 统一指标评估
    print("\n" + "="*50)
    print("[UNIFIED METRICS] Evaluating with MSE/MAE/Cosine...")
    unified_metrics = runner.validate_unified_metrics(test_loader)
    print(f"Unified Metrics:")
    print(f"  Feature MSE: {unified_metrics['mse']:.6f}")
    print(f"  Feature MAE: {unified_metrics['mae']:.6f}")
    print(f"  Cosine Similarity: {unified_metrics['cosine_sim']:.6f}")
    print(f"  Edge Loss: {unified_metrics['edge_loss']:.6f}")
    
    # 将测试结果写入日志
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== FINAL TEST RESULTS ===\n")
        lf.write(f"test_total={test_tot:.6f}\ttest_feat={test_feat:.6f}\ttest_edge={test_edge:.6f}\n")
        lf.write(f"\n=== UNIFIED METRICS ===\n")
        lf.write(f"mse={unified_metrics['mse']:.6f}\tmae={unified_metrics['mae']:.6f}\t")
        lf.write(f"cosine_sim={unified_metrics['cosine_sim']:.6f}\tedge_loss={unified_metrics['edge_loss']:.6f}\n")
    
    # 在测试集上可视化（传入output_dir）
    # 注意：visualize_results使用test_dataset（测试集），用于评估模型性能
    print("\n" + "="*50)
    print("[VIS] Generating visualizations on test set...")
    print(f"[VIS] Using test dataset with {len(test_dataset)} samples")
    visualize_results(runner, test_dataset, args, config, output_dir)


def visualize_results(runner, dataset, args, config, output_dir: Path):
    """Visualize edge maps and feature maps after training"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # 使用传入的输出目录下的visualizations子目录
    vis_output_dir = output_dir / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[VIS] Output directory: {vis_output_dir}")
    
    # 设置为评估模式
    runner.backbone.eval()
    runner.edge_head.eval()
    
    # Select first 10 images for visualization
    num_vis = min(10, len(dataset))
    indices = list(range(num_vis))
    
    print(f"[VIS] Visualizing {num_vis} images...")
    
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
            aligned_features = runner.feature_adapter(s16_features, teacher_features)
            
            # Generate edge map (使用S4特征，输出直接是256×256)
            edge_logits = runner.edge_head(s4_features, backbone_name=config.get("backbone", "yolov8"))
            # edge_logits已经是256×256，不需要上采样
            edge_pred = edge_logits
            edge_pred = torch.sigmoid(edge_pred[0, 0]).cpu().numpy()  # (256, 256)
            
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
            print(f"  [VIS] {idx+1}/{num_vis}: {save_path.name}")
    
    print(f"\nVisualization completed. Saved to: {vis_output_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()



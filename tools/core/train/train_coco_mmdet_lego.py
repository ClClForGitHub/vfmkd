#!/usr/bin/env python3
"""
A training script within the `core` workflow that leverages the MMDetection API
to fine-tune a distilled backbone on the COCO dataset.

This script demonstrates the "lego" capability by combining our custom distilled
YOLOv8Backbone with standard MMDetection components (Neck, Head) and training
it using the powerful MMDetection Runner.
"""
import os
import sys

# Set LD_LIBRARY_PATH BEFORE any imports to avoid GLIBCXX version issues
# This must be done before importing torch or any other libraries that depend on libstdc++
if hasattr(sys, 'base_prefix'):  # Running in conda environment
    conda_prefix = sys.base_prefix
    lib_path = os.path.join(conda_prefix, 'lib')
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path

import argparse
import tempfile
from pathlib import Path
import torch

# Ensure project root is in path
# project_root = Path(__file__).resolve().parents[3]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# Import custom modules FIRST to ensure they are registered before MMDetection components
# This is critical for distributed training with torchrun
try:
    import vfmkd.models.backbones.yolov8_backbone
    import vfmkd.models.necks.yolov8_pafpn
    import vfmkd.models.heads.detection.yolov8_detect_head
    print("✓ Custom VFMKD modules imported and registered")
except ImportError as e:
    print(f"Warning: Failed to import custom modules: {e}")

# MMDetection and MMYOLO components
from mmengine.config import Config
from mmengine.runner import Runner
# from mmdet.utils import setup_cache_size_limit_slurm # This is deprecated

# Define a custom hook for unfreezing the backbone
from mmengine.hooks import Hook
from mmengine.registry import HOOKS, FUNCTIONS
from mmyolo.datasets.utils import yolov5_collate

@HOOKS.register_module()
class UnfreezeBackboneHook(Hook):
    """A hook to unfreeze backbone at a specific epoch."""

    def __init__(self, unfreeze_epoch: int):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if epoch + 1 == self.unfreeze_epoch:
            # Unfreeze backbone parameters
            if hasattr(model.backbone, 'unfreeze'):
                model.backbone.unfreeze()
                print(f"Epoch {epoch + 1}: Backbone unfrozen using unfreeze() method.")
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch + 1}: Backbone unfrozen by setting requires_grad=True.")
            
            # Update optimizer: find backbone param groups and update their lr_mult
            # MMDetection's OptimWrapper uses paramwise_cfg, so we need to update the optimizer wrapper
            if hasattr(runner, 'optim_wrapper') and hasattr(runner.optim_wrapper, 'optimizer'):
                # Get the actual optimizer from the wrapper
                optimizer = runner.optim_wrapper.optimizer
                backbone_param_ids = {id(p) for p in model.backbone.parameters()}
                
                # Update learning rate for backbone parameters
                for group in optimizer.param_groups:
                    group_param_ids = {id(p) for p in group['params']}
                    if group_param_ids.intersection(backbone_param_ids):
                        # This group contains backbone parameters
                        # Update lr_mult to 1.0 (normal learning rate)
                        if 'lr_mult' in group:
                            group['lr_mult'] = 1.0
                        # Also update the actual lr if it's 0
                        if group.get('lr', 0) == 0:
                            base_lr = runner.optim_wrapper.base_lr if hasattr(runner.optim_wrapper, 'base_lr') else 0.01
                            group['lr'] = base_lr
                        print(f"Updated backbone parameter group: lr={group.get('lr', 'N/A')}, lr_mult={group.get('lr_mult', 'N/A')}")
            else:
                # Fallback: directly update optimizer if it's not wrapped
                optimizer = runner.optimizer
                backbone_param_ids = {id(p) for p in model.backbone.parameters()}
                for group in optimizer.param_groups:
                    group_param_ids = {id(p) for p in group['params']}
                    if group_param_ids.intersection(backbone_param_ids):
                        if group.get('lr', 0) == 0:
                            group['lr'] = 0.01  # Set to normal learning rate
                        print(f"Updated backbone parameter group: lr={group.get('lr', 'N/A')}")
            
            print("Backbone parameters have been updated in the optimizer with normal learning rate.")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector on COCO using MMDetection API')
    parser.add_argument(
        '--distilled-backbone', 
        help='Path to the distilled backbone checkpoint (.pth file from distillation phase)',
        default='outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth'
    )
    parser.add_argument(
        '--work-dir', 
        help='The directory to save logs and models',
        default='./work_dirs/coco_finetune_from_core_lego'
    )
    parser.add_argument(
        '--freeze-backbone', 
        action='store_true', 
        help='Freeze the backbone during training'
    )
    parser.add_argument(
        '--unfreeze-at-epoch',
        type=int,
        default=1,
        help='Epoch to unfreeze the backbone when freezing is enabled. '
             'Set <=0 to keep it frozen for the entire run.'
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=32,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--random-init',
        action='store_true',
        help='Use random initialization for backbone (no pretrained weights). Useful for baseline comparison.'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Set up environment ---
    # GPU setting: use CUDA_VISIBLE_DEVICES from environment if set, otherwise default to 5,6
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
    
    # Note: LD_LIBRARY_PATH is already set at the top of the script before imports

    # --- Prepare backbone weights (Fixed for Compatibility) ---
    tmp_backbone_path = None
    if args.random_init:
        print("=" * 60)
        print("Using RANDOM INITIALIZATION for backbone (baseline comparison)")
        print("=" * 60)
    else:
        print(f"Loading distilled backbone from: {args.distilled_backbone}")
        full_ckpt = torch.load(args.distilled_backbone, map_location='cpu')
        
        # 1. 智能识别 Key (兼容 state_dict 和 backbone)
        if 'state_dict' in full_ckpt:
            print("Detected 'state_dict' key (Clean Export format from train_distill_single_test.py).")
            backbone_state_dict = full_ckpt['state_dict']
        elif 'backbone' in full_ckpt:
            print("Detected 'backbone' key (Resume Checkpoint format).")
            backbone_state_dict = full_ckpt['backbone']
        else:
            # 最后的尝试：也许整个文件就是 state_dict
            backbone_state_dict = full_ckpt
            print("Assuming raw state dict (no wrapper keys).")

        # 2. 清洗所有可能的前缀
        new_state_dict = {}
        for k, v in backbone_state_dict.items():
            new_k = k
            
            # 2.1 清洗 torch.compile 前缀
            if new_k.startswith('_orig_mod.'):
                new_k = new_k.replace('_orig_mod.', '')
            
            # 2.2 关键修复：清洗 Wrapper 前缀 (兼容 train_distill_single_test.py 的各种输出)
            # 如果是 Resume Checkpoint，通常会有 backbone. 前缀
            # 我们需要把 backbone.backbone. 或者 backbone. 剥离掉
            # 注意：必须先处理长前缀，再处理短前缀
            if new_k.startswith('backbone.backbone.'):
                new_k = new_k.replace('backbone.backbone.', '')
            elif new_k.startswith('backbone.'):
                new_k = new_k.replace('backbone.', '')
            
            # 2.3 统一 C2f 模块命名：老版本使用 main_conv/final_conv/blocks
            if '.main_conv.' in new_k:
                new_k = new_k.replace('.main_conv.', '.cv1.')
            if '.final_conv.' in new_k:
                new_k = new_k.replace('.final_conv.', '.cv2.')
            if '.blocks.' in new_k:
                new_k = new_k.replace('.blocks.', '.m.')
            
            # 2.3 MMYOLO Backbone 需要 csp_darknet 前缀
            # 蒸馏权重里通常是 stem./stage. 开头，这里补齐
            if new_k.startswith(('stem', 'stage')):
                new_k = f'csp_darknet.{new_k}'
                
            new_state_dict[new_k] = v
        
        backbone_state_dict = new_state_dict

        # 3. 验证是否包含核心层 (Double Check)
        # 检查是否包含 'stem' 或 'stage' 等标准层，确保清洗成功
        has_stem = any('stem' in k for k in backbone_state_dict.keys())
        has_stage = any('stage' in k for k in backbone_state_dict.keys())
        if not (has_stem or has_stage):
            print("WARNING: Loaded keys do not look like a YOLO backbone (missing 'stem' or 'stage').")
            print("First 10 keys:", list(backbone_state_dict.keys())[:10])
        else:
            print(f"✓ Successfully cleaned backbone weights: {len(backbone_state_dict)} parameters")
            if has_stem:
                print("  - Contains 'stem' layers")
            if has_stage:
                print("  - Contains 'stage' layers")

        # 4. MMDetection's loader expects a file path, so we save the state dict to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(backbone_state_dict, tmp_file.name)
            tmp_backbone_path = tmp_file.name
        print(f"Saved cleaned backbone weights to: {tmp_backbone_path}")


    # --- 1. “搭乐高”：构建配置 ---
    # Base configs
    cfg = Config.fromfile('vfmkd/configs/_base_/datasets/coco_detection.py')
    cfg.merge_from_dict(Config.fromfile('vfmkd/configs/_base_/schedules/schedule_1x.py').to_dict())
    cfg.merge_from_dict(Config.fromfile('vfmkd/configs/_base_/default_runtime.py').to_dict())
    
    # Update data root to actual COCO dataset location
    cfg.data_root = '/home/team/zouzhiyuan/dataset/COCO2017'
    # Also update all dataset paths in dataloaders
    if hasattr(cfg, 'train_dataloader') and 'dataset' in cfg.train_dataloader:
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        if 'data_prefix' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset.data_prefix['img'] = 'images/train2017/'
        # Register YOLOv5 collate function into MMEngine registry (idempotent)
        try:
            FUNCTIONS.register_module(module=yolov5_collate, name='yolov5_collate')
        except KeyError:
            pass
        cfg.train_dataloader.collate_fn = dict(type='yolov5_collate')
    if hasattr(cfg, 'val_dataloader') and 'dataset' in cfg.val_dataloader:
        cfg.val_dataloader.dataset.data_root = cfg.data_root
        if 'data_prefix' in cfg.val_dataloader.dataset:
            cfg.val_dataloader.dataset.data_prefix['img'] = 'images/val2017/'
    if hasattr(cfg, 'test_dataloader') and 'dataset' in cfg.test_dataloader:
        cfg.test_dataloader.dataset.data_root = cfg.data_root
        if 'data_prefix' in cfg.test_dataloader.dataset:
            cfg.test_dataloader.dataset.data_prefix['img'] = 'images/val2017/'
    # Update evaluator paths
    if hasattr(cfg, 'val_evaluator') and 'ann_file' in cfg.val_evaluator:
        cfg.val_evaluator.ann_file = cfg.val_evaluator.ann_file.replace('data/coco', cfg.data_root)
    if hasattr(cfg, 'test_evaluator') and 'ann_file' in cfg.test_evaluator:
        cfg.test_evaluator.ann_file = cfg.test_evaluator.ann_file.replace('data/coco', cfg.data_root)
    
    # Fix visualization hooks/visualizers to use mmdet scope explicitly
    # This is needed because default_scope might be changed to 'mmyolo' by some configs
    if hasattr(cfg, 'default_hooks') and 'visualization' in cfg.default_hooks:
        cfg.default_hooks['visualization']['type'] = 'mmdet.DetVisualizationHook'
    if hasattr(cfg, 'visualizer') and isinstance(cfg.visualizer, dict):
        cfg.visualizer['type'] = 'mmdet.DetLocalVisualizer'

    # --- Modify configs for our experiment ---
    # Training settings
    cfg.train_cfg.max_epochs = 100
    cfg.train_dataloader.batch_size = args.bs
    
    # --- 官方YOLOv8标准优化器和学习率配置（覆盖schedule_1x.py的AdamW配置）---
    # 基于Ultralytics官方配置：SGD + lr0=0.01 + momentum=0.937 + weight_decay=0.0005
    # 学习率：0.01 → 0.0001 (lrf=0.01, 最终LR = lr0 * lrf)
    base_lr = 0.01  # Ultralytics官方标准lr0
    final_lr = 0.0001  # lrf = 0.01, 最终 = 0.01 * 0.01 = 0.0001
    warmup_epochs = 3.0  # Ultralytics官方标准warmup_epochs
    
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='SGD',
            lr=base_lr,
            momentum=0.937,  # Ultralytics官方标准
            weight_decay=0.0005,  # Ultralytics官方标准
            nesterov=True
        ),
        paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0)
    )
    
    # 学习率调度：Cosine退火 + Warmup（官方3 epochs）
    # Warmup: 从0.1倍开始（warmup_bias_lr=0.1），3个epoch后达到base_lr
    # Cosine: 从base_lr平滑下降到final_lr
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.1,  # warmup_bias_lr=0.1，从0.1倍开始
            by_epoch=True,
            begin=0,
            end=int(warmup_epochs),  # 3 epochs warmup
        ),
        dict(
            type='CosineAnnealingLR',
            eta_min=final_lr,  # 最终学习率 0.0001
            begin=int(warmup_epochs),  # warmup后开始
            end=100,  # 100 epochs
            T_max=100 - int(warmup_epochs),  # Cosine周期 = 97
            by_epoch=True
        )
    ]
    
    print(f"✓ 官方YOLOv8配置: SGD, lr0={base_lr}, momentum=0.937, weight_decay=0.0005")
    print(f"✓ LR调度: Warmup ({warmup_epochs} epochs, 0.1x) + CosineAnnealing ({base_lr} → {final_lr})")

    # --- Determine channel sizes (No double-scaling) ---
    width_mult_table = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.0, 'x': 1.25}
    depth_mult_table = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.0, 'x': 1.0}
    model_size = 's'
    width_mult = width_mult_table[model_size]
    depth_mult = depth_mult_table[model_size]
    c2 = int(128 * width_mult)  # S8
    c3 = int(256 * width_mult)  # S16
    c4 = int(512 * width_mult)  # S32
    actual_neck_in_channels = [c2, c3, c4]
    print(f"Configuring Neck with ACTUAL channels: {actual_neck_in_channels} (widen_factor=1.0)")

    # Model definition using YOLOv8 components
    cfg.model = dict(
        type='mmyolo.YOLODetector',
        data_preprocessor=dict(
            type='mmyolo.YOLOv5DetDataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
            bgr_to_rgb=True),
        backbone=dict(
            type='vfmkd.YOLOv8Backbone',
            model_size=model_size,
            out_indices=(1, 2, 3),
            **({'init_cfg': dict(type='Pretrained', checkpoint=tmp_backbone_path)} if tmp_backbone_path else {})
        ),
        neck=dict(
            type='mmyolo.YOLOv8PAFPN',
            in_channels=actual_neck_in_channels,
            out_channels=actual_neck_in_channels,
            deepen_factor=depth_mult,
            widen_factor=1.0,
            num_csp_blocks=3,
        ),
        bbox_head=dict(
            type='mmyolo.YOLOv8Head',
            head_module=dict(
                type='mmyolo.YOLOv8HeadModule',
                num_classes=80,
                in_channels=actual_neck_in_channels,
                widen_factor=1.0,
                reg_max=16),
            prior_generator=dict(
                type='mmdet.MlvlPointGenerator', offset=0.5,
                strides=[8, 16, 32]),
            bbox_coder=dict(type='mmyolo.DistancePointBBoxCoder'),
            # Loss settings for YOLOv8
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='none',
                loss_weight=0.5),
            loss_bbox=dict(
                type='mmyolo.IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                reduction='sum',
                loss_weight=7.5,
                return_iou=False),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=1.5)
        ),
        train_cfg=dict(
            assigner=dict(
                type='mmyolo.BatchTaskAlignedAssigner',
                num_classes=80,
                topk=10,
                alpha=0.5,
                beta=6.0)),
        test_cfg=dict(
            nms_pre=30000,
            min_bbox_size=0,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300,
            multi_label=True,
        )
    )

    # Freeze backbone if requested
    # Note: We'll freeze backbone parameters after model is built, not via frozen_stages
    # because YOLOv8Backbone doesn't implement frozen_stages yet
    if args.freeze_backbone:
        print("Backbone will be FROZEN at the start of training (via paramwise_cfg).")
        # 在官方配置基础上，添加backbone冻结设置
        # optim_wrapper已经在上面的官方配置中设置好了，这里只需要添加backbone的lr_mult=0
        if 'paramwise_cfg' not in cfg.optim_wrapper:
            cfg.optim_wrapper['paramwise_cfg'] = {}
        # 合并backbone冻结配置到现有的paramwise_cfg
        if 'custom_keys' not in cfg.optim_wrapper['paramwise_cfg']:
            cfg.optim_wrapper['paramwise_cfg']['custom_keys'] = {}
        cfg.optim_wrapper['paramwise_cfg']['custom_keys']['backbone'] = dict(lr_mult=0, decay_mult=0)
    
    # Setup unfreezing hook
    if args.freeze_backbone and args.unfreeze_at_epoch > 0:
        print(f"Backbone will be UNFROZEN at epoch {args.unfreeze_at_epoch}.")
        cfg.custom_hooks = [
            dict(type='UnfreezeBackboneHook', unfreeze_epoch=args.unfreeze_at_epoch)
        ]

    # --- 3. 运行训练 ---
    cfg.work_dir = args.work_dir
    cfg.custom_imports = dict(imports=['vfmkd'], allow_failed_imports=False)
    # Set multi-GPU training (using GPUs 5 and 6)
    # Check if running under torchrun (distributed training)
    # RANK and WORLD_SIZE are set by torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running under torchrun, use distributed training
        cfg.launcher = 'pytorch'
        cfg.find_unused_parameters = False
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print(f"Multi-GPU training detected: {world_size} GPUs (torchrun)")
    elif torch.cuda.device_count() > 1:
        # Multiple GPUs available but not using torchrun
        # Use only the first GPU to avoid distributed setup issues
        print(f"Warning: {torch.cuda.device_count()} GPUs detected but not using torchrun.")
        print("Using single GPU training. For multi-GPU, use: torchrun --nproc_per_node=2 ...")
        cfg.launcher = 'none'
        # Force to use only GPU 0 (which will be GPU 5 after CUDA_VISIBLE_DEVICES)
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '5,6').split(',')[0]
        print("Single GPU training (first GPU only)")
    else:
        cfg.launcher = 'none'
        print("Single GPU training")

    try:
        runner = Runner.from_cfg(cfg)
        runner.train()
    finally:
        # Clean up the temporary file if it exists
        if tmp_backbone_path and os.path.exists(tmp_backbone_path):
            os.remove(tmp_backbone_path)
            print(f"Cleaned up temporary backbone weights file: {tmp_backbone_path}")

if __name__ == '__main__':
    main()

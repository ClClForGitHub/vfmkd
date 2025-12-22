#!/usr/bin/env python3
"""
SAM2 头部加载与复用封装：
- 清理Hydra并按配置/权重构建SAM2
- 提取 sam_prompt_encoder / sam_mask_decoder 并设置 eval/cuda
- 返回 transforms 以便图像预处理，以及可选的 sam2_model 供实时教师特征提取
"""

import sys
from pathlib import Path
import types as _types
from typing import Tuple, Optional

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

# stub optional deps to satisfy sam2 import
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            import torch as _torch
            stub_mod.Linear = _torch.nn.Linear
        sys.modules[_mod] = stub_mod

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2 as _build_sam2
from vfmkd.sam2.sam2.utils.transforms import SAM2Transforms


def load_sam2_heads(
    device: str = 'cuda:6',
    config_file: str = 'sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path: Optional[str] = None,
    return_model: bool = True,
):
    """
    加载SAM2并返回(prompt_encoder, mask_decoder, transforms, sam2_model[可选])。
    - 统一清理Hydra实例，避免配置污染
    - 强制 mask_decoder.use_high_res_features = False
    """
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    config_dir = _ROOT / 'vfmkd' / 'sam2' / 'sam2' / 'configs'

    # 强制：Base+ 配置且权重必须存在
    if config_file != 'sam2.1/sam2.1_hiera_b+.yaml':
        raise RuntimeError('实时SAM2教师的提示/掩码编码器必须使用 Base+ 配置 (sam2.1_hiera_b+.yaml)')
    if ckpt_path is None:
        default = _ROOT / 'weights' / 'sam2.1_hiera_base_plus.pt'
        if not default.exists():
            raise RuntimeError('未找到 Base+ 权重 weights/sam2.1_hiera_base_plus.pt，请提供 --sam2-ckpt 明确路径')
        ckpt_path = str(default)
    else:
        if not Path(ckpt_path).exists():
            raise RuntimeError(f'提供的 Base+ 权重不存在: {ckpt_path}')

    # 清理Hydra并构建
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        model = _build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=str(device_obj),
        )

    model.eval()
    # 安全设置：关闭高分支特征
    if hasattr(model, 'sam_mask_decoder'):
        model.sam_mask_decoder.use_high_res_features = False

    if not hasattr(model, 'sam_prompt_encoder'):
        raise AttributeError('SAM2缺少 sam_prompt_encoder')
    if not hasattr(model, 'sam_mask_decoder'):
        raise AttributeError('SAM2缺少 sam_mask_decoder')

    prompt_encoder = model.sam_prompt_encoder.to(device_obj).eval()
    mask_decoder = model.sam_mask_decoder.to(device_obj).eval()

    # 简单统计检查，确保确实加载了权重（非空/数值合理）
    # 若存在极端情况（随机初始化），一般均值~0且std显著偏小/偏大，这里做基础守卫
    pe_first = next(prompt_encoder.parameters(), None)
    md_first = next(mask_decoder.parameters(), None)
    if pe_first is None or md_first is None:
        raise RuntimeError('Base+ 提示/掩码编码器参数为空，疑似加载失败')

    # 官方预处理
    transforms = SAM2Transforms(
        resolution=getattr(model, 'image_size', 1024),
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )

    return (prompt_encoder, mask_decoder, transforms, model) if return_model else (
        prompt_encoder,
        mask_decoder,
        transforms,
    )


__all__ = ['load_sam2_heads']



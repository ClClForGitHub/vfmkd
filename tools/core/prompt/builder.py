import sys
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter


def build_student(weights_path: str, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """构建并加载学生 backbone + adapter，返回 eval() 模式的两个模块。
    要求权重文件包含 {'backbone': state_dict, 'adapter': state_dict}
    """
    cfg = {'model_size': 's', 'pretrained': False, 'freeze_backbone': False}
    backbone = YOLOv8Backbone(cfg).to(device).eval()
    in_ch_s16 = backbone.get_feature_dims()[1]
    adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device).eval()

    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['backbone'])
    adapter.load_state_dict(ckpt['adapter'])
    return backbone, adapter


def build_sam2(device: torch.device):
    """构建 SAM2 模型（关闭高分支特征），返回 eval() 模式模型。"""
    import types as _types
    for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
        if _mod not in sys.modules:
            stub_mod = _types.ModuleType(_mod)
            if _mod == 'loralib':
                import torch as _torch
                stub_mod.Linear = _torch.nn.Linear
            sys.modules[_mod] = stub_mod

    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2 as _build

    config_dir = _ROOT / 'vfmkd' / 'sam2' / 'sam2' / 'configs'
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        sam2_model = _build(
            config_file='sam2.1/sam2.1_hiera_b+.yaml',
            ckpt_path='weights/sam2.1_hiera_base_plus.pt',
            device=str(device)
        )
    sam2_model.eval()
    sam2_model.sam_mask_decoder.use_high_res_features = False
    return sam2_model



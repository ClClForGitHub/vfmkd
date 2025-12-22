"""
YOLOv8 检测头本地实现。

参考 Ultralytics `ultralytics/nn/modules/head.py` 内 Detect 模块，保留核心行为（训练/推理）以便与官方权重兼容。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ...backbones.yolov8_components import Conv, DWConv
from ..base_head import BaseHead


def _make_anchors(
    feats: Iterable[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 anchor points 与 stride tensor，保持与 Ultralytics 原实现一致。
    返回：anchor_points (N, 2)、stride_tensor (N, 1)。
    """

    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def _dist2bbox(distance: torch.Tensor, anchors: torch.Tensor, xywh: bool = True, dim: int = 1) -> torch.Tensor:
    """
    将距离分布解码为 bbox，来源于 Ultralytics `dist2bbox`。
    """

    lt, rb = distance.split((2, 2), dim)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class DistributionFocalLoss(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL)。
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1)
        self.conv.weight.data[:] = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, hw = x.shape
        x = x.view(b, 4, self.c1, hw).softmax(2)  # [B,4,reg_max,HW]
        x = x.mul(torch.arange(self.c1, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)).sum(2)
        return x.view(b, 4, hw)


@dataclass
class DetectHeadConfig:
    num_classes: int = 80
    reg_max: int = 16
    legacy: bool = False
    end2end: bool = False
    max_det: int = 300

@MODELS.register_module()
class YOLOv8DetectHead(BaseModule):
    """
    Localized YOLOv8 detection head, adapted for MMDetection.
    """

    def __init__(self, 
                 num_classes: int = 80,
                 in_channels: Sequence[int] = (128, 256, 512),
                 reg_max: int = 16,
                 init_cfg: Optional[Dict] = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_max = reg_max

        # The following attributes are adapted from the original __init__
        # to be compatible with MMDetection's config-driven approach.
        self.nl = len(in_channels)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        
        c2 = max(16, self.in_channels[0] // 4, self.reg_max * 4)
        c3 = max(self.in_channels[0], min(self.num_classes, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(ch, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for ch in self.in_channels
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(ch, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1))
            for ch in self.in_channels
        )

        self.dfl = DistributionFocalLoss(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Forward features from the neck."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x
    
    # The custom inference logic (_inference, set_strides, bias_init) is removed.
    # MMDetection's post-processing components (e.g., AnchorFreeHead) will handle this,
    # and initialization is managed via `init_cfg`. Our module's responsibility is now
    # purely the forward pass of the convolution layers.


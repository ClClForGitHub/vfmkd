"""
YOLOv8 PA-FPN 实现，适配多尺度知识蒸馏训练。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet.registry import MODELS

from ..backbones.yolov8_components import C2f, Conv


def _make_divisible(value: float, divisor: int = 8) -> int:
    """
    将通道数调整为可整除的数值（参考 YOLO 实现）。
    """

    min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

@MODELS.register_module()
class YOLOv8PAFPN(BaseModule):
    """
    基于 YOLOv8 的 Path Aggregation FPN。

    输入特征默认顺序：[S4, S8, S16, S32]。
    输出特征顺序：[P3 (S8), P4 (S16), P5 (S32)]。
    """

    # [depth_mult, width_mult, max_channels]
    _SCALE_TABLE: Dict[str, Tuple[float, float, int]] = {
        "n": (0.33, 0.25, 1024),
        "s": (0.33, 0.50, 1024),
        "m": (0.67, 0.75, 768),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.25, 512),
    }

    _BASE_DET_CHANNELS: Tuple[int, int, int] = (256, 512, 1024)
    _BASE_DEPTHS: Tuple[int, int, int, int] = (3, 3, 3, 3)

    def __init__(self, 
                 model_size: str = 's',
                 in_channels: Optional[Sequence[int]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        self.model_size: str = model_size.lower()
        scale_cfg = self._SCALE_TABLE.get(self.model_size)
        if scale_cfg is None:
            raise ValueError(f"Unsupported model_size: {self.model_size}")

        depth_mult, width_mult, max_channels = scale_cfg
        
        if in_channels is None:
            in_channels = self._default_in_channels(self.model_size)
        
        if len(in_channels) != 4:
            raise ValueError("YOLOv8PAFPN expects 4 in_channels ([S4, S8, S16, S32]).")

        self.in_channels = tuple(int(c) for c in in_channels)

        # 根据 width_mult 计算输出通道
        base = self._BASE_DET_CHANNELS
        self.out_channels = tuple(
            min(max_channels, _make_divisible(c * width_mult)) for c in base
        )  # P3, P4, P5
        
        # 修正：当未提供in_channels时，根据backbone的真实配置动态计算
        if not in_channels:
             # 从backbone的配置动态推断
            bb_w_mult = self._SCALE_TABLE[self.model_size][1]
            c2 = _make_divisible(128 * bb_w_mult)
            c3 = _make_divisible(256 * bb_w_mult)
            c4 = _make_divisible(512 * bb_w_mult)
            # S4和S8通道相同
            in_channels = (c2, c2, c3, c4)

        # 根据 depth_mult 计算各阶段重复次数
        base_depth = self._BASE_DEPTHS
        self.depths = tuple(max(int(round(d * depth_mult)), 1) for d in base_depth)

        # 构建模块
        c_s4, c_s8, c_s16, c_s32 = self.in_channels
        p3_c, p4_c, p5_c = self.out_channels
        d1, d2, d3, d4 = self.depths

        self.reduce_conv_p5 = Conv(c_s32, p4_c, k=1, s=1)
        self.c2f_p4 = C2f(c_s16 + p4_c, p4_c, n=d1, shortcut=False)

        self.reduce_conv_p4 = Conv(p4_c, p3_c, k=1, s=1)
        self.c2f_p3 = C2f(c_s8 + p3_c, p3_c, n=d2, shortcut=False)

        self.down_conv_p3 = Conv(p3_c, p4_c, k=3, s=2)
        self.c2f_out_p4 = C2f(p4_c + p4_c, p4_c, n=d3, shortcut=False)

        self.down_conv_p4 = Conv(p4_c, p5_c, k=3, s=2)
        self.c2f_out_p5 = C2f(p5_c + c_s32, p5_c, n=d4, shortcut=False)

    @staticmethod
    def _default_in_channels(model_size: str) -> Tuple[int, int, int, int]:
        """
        当未显式提供 in_channels 时，根据 backbone 规格给出默认值。
        """

        defaults: Dict[str, Tuple[int, int, int, int]] = {
            "n": (64, 128, 256, 512),
            "s": (64, 128, 256, 512),
            "m": (96, 192, 384, 768),
            "l": (128, 256, 512, 1024),
            "x": (160, 320, 640, 1280),
        }
        if model_size not in defaults:
            raise ValueError(f"未找到 model_size={model_size} 对应的默认 in_channels。")
        # BackBone forward 返回 [S4, S8, S16, S32]，其中 S4 与 S8 通道相等
        c2 = defaults[model_size][1]
        c3 = defaults[model_size][2]
        c4 = defaults[model_size][3]
        return (defaults[model_size][1], c2, c3, c4)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        if len(features) < 4:
            # 兼容只输入3个特征的情况 [S8,S16,S32]，此时认为S4=S8
            if len(features) == 3:
                s8, s16, s32 = features
                s4 = s8
            else:
                raise ValueError("YOLOv8PAFPN 需要至少 3 个输入特征（[S8, S16, S32]）。")
        else:
            s4, s8, s16, s32 = features[-4:]

        # 上采样路径
        # 注意：官方实现中，这里s32直接作为p5_out的输入之一，而上采样用的是一个1x1卷积降维后的结果
        p5_reduced = self.reduce_conv_p5(s32)
        p5_upsampled = F.interpolate(p5_reduced, scale_factor=2.0, mode="nearest")
        p4 = self.c2f_p4(torch.cat([p5_upsampled, s16], dim=1))

        p4_reduced = self.reduce_conv_p4(p4)
        p4_upsampled = F.interpolate(p4_reduced, scale_factor=2.0, mode="nearest")
        p3 = self.c2f_p3(torch.cat([p4_upsampled, s8], dim=1))

        # 下采样路径
        p3_down = self.down_conv_p3(p3)
        p4_out = self.c2f_out_p4(torch.cat([p3_down, p4], dim=1))

        p4_down = self.down_conv_p4(p4_out)
        p5_out = self.c2f_out_p5(torch.cat([p4_down, s32], dim=1))

        return p3, p4_out, p5_out

    def get_out_channels(self) -> List[int]:
        return list(self.out_channels)


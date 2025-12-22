"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modified for Bio-Retina Injection (RT-DETRv2)
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register
# ============ [新增] 导入 BioRetina 模块 ============
from .bio_retina import BioRetinaModule
# ===================================================


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module,
        use_dog: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_dog = use_dog
        
        # ============ [新增] 初始化 BioRetinaModule（可选） ============
        # 默认不启用 DoG 增强，只有在 use_dog=True 时才构建与使用
        self.bio_retina = None
        if self.use_dog:
            # 自动探测 Backbone 的通道数和步长，适配 HGNetv2 或 ResNet
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = [512, 1024, 2048]  # 默认 ResNet 兜底

            if hasattr(backbone, 'out_strides'):
                out_strides = backbone.out_strides
            else:
                out_strides = [8, 16, 32]

            print(f"[BioRetina] Injected into RT-DETRv2! Channels: {out_channels}, Strides: {out_strides}")

            # 针对小目标优化的参数: sigma1=1.0 (3x3像素), sigma2=2.0
            self.bio_retina = BioRetinaModule(
                strides=out_strides,
                in_channels=out_channels,
                sigma1=1.0,
                sigma2=2.0,
                dog_threshold=0.0  # 让网络自己去学阈值
            )
        # ==========================================================
        
    def forward(self, x, targets=None):
        # 1. Backbone 提取特征 [S3, S4, S5]
        # 保存一下原图 x，用于计算 DoG
        images = x 
        x = self.backbone(x)
        
        # ============ [新增] 可选：注入 DoG 特征 ============
        # 在进入 Encoder 之前，把原图的 DoG 边缘信息注入到 backbone 特征中
        # 只有在 use_dog=True 且已构建 bio_retina 时才执行
        if self.bio_retina is not None:
            # 输入：images(原图), x(特征列表)
            # 输出：增强后的特征列表
            x = self.bio_retina(images, x)
        # =================================================

        # 2. Encoder (HybridEncoder: AIFI + CCFM)
        # 此时 S5 已被 DoG 增强，AIFI 自注意力会利用这些高频信息
        x = self.encoder(x)        
        
        # 3. Decoder
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
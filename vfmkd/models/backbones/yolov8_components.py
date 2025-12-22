"""
YOLOv8 core components implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv(nn.Module):
    """
    Standard convolution with BatchNorm and activation.
    """
    
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        """
        Initialize Conv layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            act: Whether to use activation
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self._autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))
    
    def _autopad(self, k: int, p: Optional[int]) -> int:
        """Auto padding."""
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class Bottleneck(nn.Module):
    """
    Standard bottleneck block.
    """
    
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, 
                 k: Tuple[int, int] = (3, 3), e: float = 0.5):
        """
        Initialize Bottleneck.
        
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Whether to use shortcut
            g: Groups
            k: Kernel sizes
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.
    """
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, 
                 g: int = 1, e: float = 0.5):
        """
        Initialize C2f.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Whether to use shortcut
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.
    """
    
    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize SPPF.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone for YOLOv8.
    """
    
    def __init__(self, model_size: str = 'n'):
        """
        Initialize CSPDarknet.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
        """
        super().__init__()
        
        # Model configurations
        configs = {
            'n': [64, 128, 256, 512, 1024],  # [c1, c2, c3, c4, c5]
            's': [64, 128, 256, 512, 1024],
            'm': [96, 192, 384, 768, 1536],
            'l': [128, 256, 512, 1024, 2048],
            'x': [160, 320, 640, 1280, 2560],
        }
        
        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")
        
        channels = configs[model_size]
        
        # Stem
        self.stem = Conv(3, channels[0], k=3, s=2, p=1)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], k=3, s=2, p=1),
            C2f(channels[1], channels[1], n=1, shortcut=True)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], k=3, s=2, p=1),
            C2f(channels[2], channels[2], n=2, shortcut=True)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], k=3, s=2, p=1),
            C2f(channels[3], channels[3], n=2, shortcut=True)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], k=3, s=2, p=1),
            C2f(channels[4], channels[4], n=1, shortcut=True),
            SPPF(channels[4], channels[4], k=5)
        )
        
        # Store channel info for feature extraction
        self.channels = channels
        self.model_size = model_size
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through CSPDarknet.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of feature tensors at different scales:
            [feat_s4, feat_s8, feat_s16, feat_s32]
            - feat_s4: 4x downsampled (256×256 for 1024 input)
            - feat_s8: 8x downsampled (128×128 for 1024 input)
            - feat_s16: 16x downsampled (64×64 for 1024 input)
            - feat_s32: 32x downsampled (32×32 for 1024 input)
        """
        # Stem
        x = self.stem(x)  # [B, c1, H/2, W/2] = 512×512
        
        # Stage 1 - 提取S4特征（4倍下采样，256×256）
        x = self.stage1[0](x)  # [B, c2, H/4, W/4] = 256×256
        feat_s4 = x  # S4: 4x downsampled
        
        x = self.stage1[1](x)  # [B, c2, H/4, W/4]
        feat_s8 = x  # S8: 8x downsampled
        
        # Stage 2
        x = self.stage2[0](x)  # [B, c3, H/8, W/8]
        x = self.stage2[1](x)  # [B, c3, H/8, W/8]
        feat_s16 = x  # S16: 16x downsampled
        
        # Stage 3
        x = self.stage3[0](x)  # [B, c4, H/16, W/16]
        x = self.stage3[1](x)  # [B, c4, H/16, W/16]
        feat_s32 = x  # S32: 32x downsampled
        
        # Stage 4
        x = self.stage4[0](x)  # [B, c5, H/32, W/32]
        x = self.stage4[1](x)  # [B, c5, H/32, W/32]
        x = self.stage4[2](x)  # [B, c5, H/32, W/32] - SPPF
        
        return [feat_s4, feat_s8, feat_s16, feat_s32]
    
    def get_feature_dims(self) -> list:
        """Get feature dimensions."""
        # 返回 [S4, S8, S16, S32] 的通道数
        # S4和S8都是channels[1]，S16是channels[2]，S32是channels[3]
        return [self.channels[1], self.channels[1], self.channels[2], self.channels[3]]
    
    def get_feature_strides(self) -> list:
        """Get feature strides."""
        # 返回 [S4, S8, S16, S32] 的下采样倍数
        return [4, 8, 16, 32]

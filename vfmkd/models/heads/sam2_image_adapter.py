from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
_sam2_path = _project_root / "vfmkd" / "sam2"
if str(_sam2_path) not in sys.path:
    sys.path.insert(0, str(_sam2_path))
try:
    from sam2.modeling.sam2_utils import LayerNorm2d
except ImportError:
    # 如果sam2不可用，使用PyTorch的LayerNorm作为替代
    import torch.nn as nn
    LayerNorm2d = nn.LayerNorm
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class Sam2ImageAdapter(BaseModule):
    """
    Adapter to align student features with SAM2 teacher features.
    This adapter is used to bridge the gap between student and teacher feature spaces.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 256, **kwargs):
        """
        Initialize the SAM2 Image Adapter.
        
        Args:
            in_channels: Number of input channels from student backbone
            out_channels: Number of output channels (should match SAM2 feature channels, typically 256)
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1 convolution to adjust channel dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Layer normalization for feature alignment
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Input tensor of shape (B, C, H, W) from student backbone
            
        Returns:
            Aligned feature tensor of shape (B, out_channels, H, W)
        """
        # Channel adjustment
        x = self.conv(x)
        
        # Normalization
        x = self.norm(x)
        
        return x

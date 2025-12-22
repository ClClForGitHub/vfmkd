"""
YOLOv8 backbone implementation for VFMKD.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Sequence

from mmengine.model import BaseModule
from mmdet.registry import MODELS

from .yolov8_components import CSPDarknet

@MODELS.register_module()
class YOLOv8Backbone(BaseModule):
    """
    YOLOv8 backbone implementation for VFMKD, adapted for MMDetection.
    
    This backbone implements the CSPDarknet architecture used in YOLOv8.
    """
    
    def __init__(
        self,
        model_size: str = 's',
        init_cfg: Dict = None,
        out_indices: Optional[Sequence[int]] = None,
        **kwargs):
        """
        Initialize YOLOv8 backbone.
        
        Args:
            model_size (str): Size of the model ('n', 's', 'm', 'l', 'x').
            init_cfg (dict, optional): Initialization config dict.
        """
        super().__init__(init_cfg)
        
        self.model_size = model_size
        
        # Initialize CSPDarknet
        self.csp_darknet = CSPDarknet(model_size=self.model_size)
        
        # Get feature information
        self.feature_dims = self.csp_darknet.get_feature_dims()
        self.feature_strides = self.csp_darknet.get_feature_strides()

        total_levels = len(self.feature_dims)
        if out_indices is None:
            out_indices = tuple(range(total_levels))

        if not isinstance(out_indices, (tuple, list)):
            raise TypeError("out_indices must be a tuple or list of integers.")
        if len(out_indices) == 0:
            raise ValueError("out_indices must contain at least one feature index.")

        validated_indices = []
        for idx in out_indices:
            if not isinstance(idx, int):
                raise TypeError("All entries in out_indices must be integers.")
            if idx < 0 or idx >= total_levels:
                raise ValueError(
                    f"out_indices contains invalid level {idx}. "
                    f"Valid range: [0, {total_levels - 1}].")
            if idx not in validated_indices:
                validated_indices.append(idx)

        self.out_indices = tuple(validated_indices)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass through YOLOv8 backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of feature tensors at different scales. MMDetection necks
            typically expect a tuple.
        """
        feats = self.csp_darknet(x)
        selected = [feats[i] for i in self.out_indices]
        return tuple(selected)
    
    def get_feature_dims(self) -> List[int]:
        """
        Get the number of channels for each feature level.
        """
        return [self.feature_dims[i] for i in self.out_indices]
    
    def get_feature_strides(self) -> List[int]:
        """
        Get the stride (downsampling factor) for each feature level.
        """
        return [self.feature_strides[i] for i in self.out_indices]

# Note: The BaseBackbone inheritance was removed as per MMDetection 3.x practices,
# where backbones can inherit directly from mmengine.model.BaseModule.
# The custom weight loading logic (_load_external_backbone_weights) is removed
# as this will now be handled declaratively by MMDetection's `init_cfg`.

"""
Backbone implementations for VFMKD.
"""

from .base_backbone import BaseBackbone
from .repvit_backbone import RepViTBackbone
from .yolov8_backbone import YOLOv8Backbone

# TODO: Implement other backbones
# from .vit_backbone import ViTBackbone
# from .mamba_backbone import MambaBackbone

__all__ = [
    "BaseBackbone",
    "RepViTBackbone",
    "YOLOv8Backbone",
    # "ViTBackbone", 
    # "MambaBackbone",
]
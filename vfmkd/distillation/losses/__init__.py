"""
Distillation loss functions for VFMKD.
"""

from .feature_loss import FeatureLoss
from .detection_loss import DetectionLoss
from .segmentation_loss import SegmentationLoss

__all__ = [
    "FeatureLoss",
    "DetectionLoss", 
    "SegmentationLoss",
]
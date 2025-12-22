from .base_head import BaseHead
from .sam2_adapter_head import SAM2SegAdapterHead

__all__ = [
    "BaseHead",
    "SAM2SegAdapterHead",
]

"""
Head implementations for VFMKD.
"""

from .base_head import BaseHead

# TODO: Implement other heads
# from .yolo_head import YOLOHead
# from .detr_head import DETRHead
# from .sam_head import SAMHead

__all__ = [
    "BaseHead",
    # "YOLOHead",
    # "DETRHead", 
    # "SAMHead",
]
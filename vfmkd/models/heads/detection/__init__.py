"""
Detection head implementations.
"""

from typing import Dict, Optional, Type

# BaseHead 可能不存在，先尝试导入，如果失败则跳过
try:
    from ..base_head import BaseHead
except ImportError:
    BaseHead = None

from .yolov8_detect_head import YOLOv8DetectHead

# Registry for detection heads
_DETECTION_HEAD_REGISTRY: Dict[str, Type] = {
    'yolov8_detect_head': YOLOv8DetectHead,
}


def register_detection_head(name: str, head_cls: Type) -> None:
    """Register a detection head class."""
    key = name.lower()
    if key in _DETECTION_HEAD_REGISTRY:
        raise ValueError(f"Detection head '{name}' already registered.")
    _DETECTION_HEAD_REGISTRY[key] = head_cls


def get_detection_head_registry() -> Dict[str, Type]:
    """Get the detection head registry."""
    return dict(_DETECTION_HEAD_REGISTRY)


def build_detection_head(name: str, config: Optional[dict] = None) -> Type:
    """Build a detection head by name."""
    key = name.lower()
    if key not in _DETECTION_HEAD_REGISTRY:
        raise KeyError(f"Unknown detection head '{name}'. Available: {list(_DETECTION_HEAD_REGISTRY.keys())}")
    return _DETECTION_HEAD_REGISTRY[key]


__all__ = [
    'YOLOv8DetectHead',
    'register_detection_head',
    'get_detection_head_registry',
    'build_detection_head',
]

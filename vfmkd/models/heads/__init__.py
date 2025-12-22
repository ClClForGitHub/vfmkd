"""
Head 实现与工厂函数。
"""

from typing import Dict, Optional, Type

# BaseHead 可能不存在，先尝试导入，如果失败则跳过
try:
    from .base_head import BaseHead
except ImportError:
    BaseHead = None
from .detection import (
    YOLOv8DetectHead,
    build_detection_head,
    get_detection_head_registry,
    register_detection_head,
)
from .detection.yolov8_detect_head import YOLOv8DetectHead
from .sam2_image_adapter import Sam2ImageAdapter

# SAM2 适配器头（需要 sam2 模块，可选导入）
try:
    from .sam2_adapter_head import SAM2SegAdapterHead

    _SAM2_AVAILABLE = True
except ImportError:  # pragma: no cover - sam2 为可选依赖
    SAM2SegAdapterHead = None
    _SAM2_AVAILABLE = False

__all__ = ['YOLOv8DetectHead', 'Sam2ImageAdapter']

if _SAM2_AVAILABLE:
    __all__.append("SAM2SegAdapterHead")


_HEAD_REGISTRY: Dict[str, Type[BaseHead]] = {}
_HEAD_REGISTRY.update(get_detection_head_registry())

if _SAM2_AVAILABLE and SAM2SegAdapterHead is not None:
    _HEAD_REGISTRY["sam2_adapter_head"] = SAM2SegAdapterHead


def register_head(name: str, head_cls: Type[BaseHead]) -> None:
    key = name.lower()
    if key in _HEAD_REGISTRY:
        raise ValueError(f"Head '{name}' 已存在。")
    _HEAD_REGISTRY[key] = head_cls


def build_head(name: str, config: Optional[dict] = None) -> BaseHead:
    if not name:
        raise ValueError("构建 Head 时必须提供名称。")
    key = name.lower()
    if key not in _HEAD_REGISTRY:
        raise KeyError(f"未知的 Head '{name}'，可用项：{list(_HEAD_REGISTRY.keys())}")
    return _HEAD_REGISTRY[key](config or {})


def get_head_registry() -> Dict[str, Type[BaseHead]]:
    return dict(_HEAD_REGISTRY)

"""
简化的 BaseHead 抽象基类，避免在训练学生蒸馏（不使用检测头）时的硬依赖。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn


class BaseHead(nn.Module):
    """
    轻量占位实现：
    - 提供通用的 forward 接口占位
    - 下游检测头可继承并覆盖
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

    def forward(self, *args, **kwargs):  # pragma: no cover - 仅占位
        raise NotImplementedError("BaseHead.forward should be implemented by subclasses.")



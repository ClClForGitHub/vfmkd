"""
VFMKD - Vision Foundation Model Knowledge Distillation

A framework for knowledge distillation from vision foundation models.
"""

__version__ = "0.1.0"
__author__ = "VFMKD Team"
__email__ = "vfmkd@example.com"

# Core imports
from .models import *
from .teachers import *
from .distillation import *
# from .datasets import *  # 模块不存在，暂时注释
from .utils import *
from .core import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
]
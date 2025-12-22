"""
VFMKD - Vision Foundation Model Knowledge Distillation

A framework for knowledge distillation from vision foundation models.
"""

__version__ = "0.1.0"
__author__ = "VFMKD Team"
__email__ = "vfmkd@example.com"

# Core imports to trigger registration
from .models import *
# from .teachers import * # Assuming these are not needed for registration for now
# from .distillation import *
# from .utils import *
# from .core import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
] + models.__all__ # Add all models to the top-level namespace
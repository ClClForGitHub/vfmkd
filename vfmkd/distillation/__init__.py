"""
Distillation modules for VFMKD.
"""

# TODO: Implement distillation modules
# from .offline_distiller import OfflineDistiller
# from .online_distiller import OnlineDistiller
# from .feature_storage import FeatureStorage
from .losses import *
from .adapters import SimpleAdapter, EdgeAdapter

__all__ = [
    # "OfflineDistiller",
    # "OnlineDistiller", 
    # "FeatureStorage",
    "SimpleAdapter",
    "EdgeAdapter",
]
"""
Teacher model implementations for VFMKD.
"""

from .base_teacher import BaseTeacher
from .sam2_teacher import SAM2Teacher

# TODO: Implement other teachers
# from .dino_teacher import DINOTeacher
# from .teacher_manager import TeacherManager

try:
    from .dinov3_teacher import DinoV3Teacher
    __all__ = [
        "BaseTeacher",
        "SAM2Teacher",
        "DinoV3Teacher",
        # "DINOTeacher",
        # "TeacherManager",
    ]
except ImportError:
    # DinoV3 teacher 可能还未 vendor，暂时不导出
    __all__ = [
        "BaseTeacher",
        "SAM2Teacher",
        # "DinoV3Teacher",
        # "DINOTeacher",
        # "TeacherManager",
    ]
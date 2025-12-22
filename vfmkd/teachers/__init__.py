"""
Teacher model implementations for VFMKD.
"""

from .base_teacher import BaseTeacher
from .sam2_teacher import SAM2Teacher

# TODO: Implement other teachers
# from .dino_teacher import DINOTeacher
# from .teacher_manager import TeacherManager

__all__ = [
    "BaseTeacher",
    "SAM2Teacher",
    # "DINOTeacher",
    # "TeacherManager",
]
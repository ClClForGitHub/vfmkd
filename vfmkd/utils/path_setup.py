import os
import sys
from typing import Optional


def ensure_project_paths(project_root: Optional[str] = None) -> None:
	"""
	确保项目根目录与本地 sam2 包父目录加入 sys.path。
	- project_root: 可显式传入项目根；默认从本文件相对定位两级目录。
	"""
	if project_root is None:
		project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	# 将 vfmkd/sam2 的父目录加入，保证子模块中的 "import sam2" 可用
	local_sam2_parent = os.path.join(project_root, 'vfmkd', 'sam2')
	if local_sam2_parent not in sys.path:
		sys.path.insert(0, local_sam2_parent)




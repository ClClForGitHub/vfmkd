from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root / "vfmkd" / "sam2") not in sys.path:
    sys.path.insert(0, str(_project_root / "vfmkd" / "sam2"))
from sam2.modeling.sam2_utils import LayerNorm2d


class Sam2ImageAdapter(nn.Module):
    """
    EdgeSAM风格的对齐适配器：1x1 Conv → LayerNorm2d → 3x3 Conv → LayerNorm2d。
    - 输入：来自backbone的 [s8, s16, s32] 特征列表
    - 仅使用 s16 分支进行对齐到 SAM2 的 image_embeddings 空间
    - 输出：张量 (B, 256, 64, 64)
    """

    def __init__(self, in_channels_s16: int, image_size: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.target_hw = image_size // 16  # 64 when image_size=1024

        self.proj = nn.Conv2d(in_channels_s16, hidden_dim, kernel_size=1, bias=False)
        self.ln1 = LayerNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.ln2 = LayerNorm2d(hidden_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        assert len(features) >= 2, "需要 [s8, s16, s32] 中至少到 s16"
        feat_s16 = features[1]
        x = self.proj(feat_s16)
        x = self.ln1(x)
        x = self.conv3(x)
        x = self.ln2(x)
        if x.shape[-1] != self.target_hw or x.shape[-2] != self.target_hw:
            x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return x



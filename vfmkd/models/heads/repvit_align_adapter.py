from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root / "vfmkd" / "sam2") not in sys.path:
    sys.path.insert(0, str(_project_root / "vfmkd" / "sam2"))
from sam2.modeling.sam2_utils import LayerNorm2d


class RepViTAlignAdapter(nn.Module):
    """
    对 RepViT 最终特征做最小对齐：LN → 3x3 Conv(256→256) → LN，输出 (B,256,64,64)。
    不做通道对齐（假定 RepViT 已输出 256 通道）。
    """

    def __init__(self, image_size: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.target_hw = image_size // 16  # 64

        self.ln1 = LayerNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.ln2 = LayerNorm2d(hidden_dim)

    def forward(self, features: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        x = features[-1] if isinstance(features, list) else features
        x = self.ln1(x)
        x = self.conv3(x)
        x = self.ln2(x)
        if x.shape[-1] != self.target_hw or x.shape[-2] != self.target_hw:
            x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return x



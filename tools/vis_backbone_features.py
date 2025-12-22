import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# 项目根入路径
_ROOT = Path(__file__).parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone


def load_image_rgb(path: str, size: int = 1024) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size))
    x = TF.to_tensor(img).unsqueeze(0)
    return x


def to_energy_map(feat_bchw: torch.Tensor) -> torch.Tensor:
    x = feat_bchw[0]  # [C,H,W]
    energy = torch.sqrt(torch.clamp((x * x).sum(dim=0), min=0) + 1e-12)  # [H,W]
    # 归一化到[0,1]
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
    return energy


def grid_channels(feat_bchw: torch.Tensor, max_channels: int = 16) -> torch.Tensor:
    x = feat_bchw[0]  # [C,H,W]
    c = min(x.shape[0], max_channels)
    x = x[:c]
    # 归一化每个通道到[0,1]
    x = x - x.amin(dim=(1, 2), keepdim=True)
    den = (x.amax(dim=(1, 2), keepdim=True) - x.amin(dim=(1, 2), keepdim=True) + 1e-6)
    x = x / den
    # 排成网格
    grid_cols = int(max(1, c**0.5))
    grid_rows = (c + grid_cols - 1) // grid_cols
    h, w = x.shape[1], x.shape[2]
    canvas = torch.zeros((grid_rows * h, grid_cols * w), dtype=x.dtype, device=x.device)
    for i in range(c):
        r = i // grid_cols
        col = i % grid_cols
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = x[i]
    return canvas


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--yolov8_weight', type=str, default='')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_dir', type=str, default='outputs')
    args = ap.parse_args()

    # 设备优先CUDA
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # 构建backbone并尝试部分加载权重
    backbone = YOLOv8Backbone({
        'model_size': 's',
        'pretrained': False,
        'external_weight_path': args.yolov8_weight if args.yolov8_weight else None,
    }).to(device)
    backbone.eval()

    # 前向
    x = load_image_rgb(args.image, size=1024).to(device)
    feats = backbone(x)  # [s8, s16, s32]

    # 输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化：每层能量图与通道网格
    names = ['s8', 's16', 's32']
    for f, name in zip(feats, names):
        # 统一resize到1024便于观察
        energy = to_energy_map(f)
        energy_up = F.interpolate(energy[None, None], size=(1024, 1024), mode='bilinear', align_corners=False)[0, 0]
        grid = grid_channels(f, max_channels=16)
        grid_up = F.interpolate(grid[None, None], size=(1024, 1024), mode='nearest')[0, 0]

        # 保存PNG
        Image.fromarray((energy_up.clamp(0, 1).cpu().numpy() * 255).astype('uint8')).save(out_dir / f'yolov8_backbone_{name}_energy.png')
        Image.fromarray((grid_up.clamp(0, 1).cpu().numpy() * 255).astype('uint8')).save(out_dir / f'yolov8_backbone_{name}_grid.png')

    print(f"Saved backbone feature visualizations to {out_dir}")


if __name__ == '__main__':
    main()



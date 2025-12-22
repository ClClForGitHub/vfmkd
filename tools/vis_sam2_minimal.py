import argparse
import os
import sys
import json
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

# 保证脚本可直接运行：将项目根和本地 sam2 包父目录加入sys.path
# 在导入vfmkd之前，先手动注入路径，避免因导入顺序导致的循环依赖
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SAM2_PARENT = os.path.join(_ROOT, 'vfmkd', 'sam2')
if _SAM2_PARENT not in sys.path:
    sys.path.insert(0, _SAM2_PARENT)

# 直接设置路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "vfmkd" / "sam2"))

# 将本地 sam2 包注册为顶级别名，确保 'from sam2.*' 可用
try:
    import importlib
    sam2_pkg = importlib.import_module('vfmkd.sam2.sam2')
    sys.modules.setdefault('sam2', sam2_pkg)
except Exception:
    pass

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_adapter_head import SAM2SegAdapterHead


def load_image_rgb(path: str, size: int = 1024) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    x = TF.to_tensor(img)  # [3,H,W], [0,1]
    x = x.unsqueeze(0)
    return x


def load_mask_from_json(json_path: str, img_size: int = 1024) -> torch.Tensor:
    """
    期望JSON中包含 polygon 或 rle 或 直接二值掩码路径；这里做最小支持：
    - 若有 'mask_path': 直接读图并resize到 img_size；
    - 若有 'points': [(x,y), ...] polygon，使用PIL绘制为mask；
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mask = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    if 'mask_path' in data and os.path.isfile(data['mask_path']):
        m = Image.open(data['mask_path']).convert('L').resize((img_size, img_size))
        return TF.to_tensor(m).unsqueeze(0)  # [1,1,H,W]
    if 'points' in data and isinstance(data['points'], list) and len(data['points']) >= 3:
        pts = [(float(x), float(y)) for x, y in data['points']]
        draw.polygon(pts, fill=255)
        return TF.to_tensor(mask).unsqueeze(0)
    # fallback：全零
    return TF.to_tensor(mask).unsqueeze(0)


def sample_foreground_point_from_mask(mask_1x1_hw: torch.Tensor) -> tuple:
    """mask: [1,1,H,W] in {0,1} or [0,255]. 返回 (x,y)。"""
    m = (mask_1x1_hw > 0.5).float()
    ys, xs = torch.where(m[0, 0] > 0.5)
    if ys.numel() == 0:
        # 无前景时取中心
        return (512.0, 512.0)
    idx = random.randrange(ys.numel())
    y = ys[idx].item()
    x = xs[idx].item()
    return (float(x), float(y))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--out', type=str, default='sam2_minimal_mask.png')
    parser.add_argument('--yolov8_weight', type=str, default='')
    parser.add_argument('--point', type=float, nargs=2, default=[-1, -1])
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--sa1b_json', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--multimask', type=int, default=3)
    args = parser.parse_args()

    # 直接尊重用户指定设备（若不可用将报错，便于发现问题）
    device = torch.device(args.device)

    # Backbone
    backbone = YOLOv8Backbone({
        'model_size': 'n',
        'pretrained': False,
        'external_weight_path': args.yolov8_weight if args.yolov8_weight else None,
    }).to(device)

    # Head
    in_dims = backbone.get_feature_dims()
    head = SAM2SegAdapterHead({
        'image_size': 1024,
        'hidden_dim': 256,
        'in_channels': in_dims,
        'backbone_strides': backbone.get_feature_strides(),
        'use_high_res_features': False,
        'num_multimask_outputs': max(1, int(args.multimask)),
    }).to(device)

    # Image
    x = load_image_rgb(args.image, size=1024).to(device)

    # Forward
    feats = backbone(x)
    # 根据JSON/掩码随机采样前景点；若未提供则使用传入的 --point
    if args.sa1b_json and os.path.isfile(args.sa1b_json):
        mask_hint = load_mask_from_json(args.sa1b_json, img_size=1024)  # [1,1,1024,1024]
        px, py = sample_foreground_point_from_mask(mask_hint)
        point_xy = [px, py]
    else:
        point_xy = args.point
        if point_xy[0] < 0 or point_xy[1] < 0:
            point_xy = [512.0, 512.0]

    pts = torch.tensor([[point_xy]], device=device, dtype=torch.float32)
    lbl = torch.tensor([[args.label]], device=device, dtype=torch.int32)
    out = head(
        feats,
        point_coords=pts,
        point_labels=lbl,
        num_multimask_outputs=max(1, int(args.multimask)),
    )

    logits = out['low_res_logits']  # [B,M,256,256] when multimask>1 (logits at 1/4 scale)
    ious = out.get('iou_predictions', None)
    # 选择IoU最高的掩码
    if logits.dim() == 4 and logits.shape[1] > 1 and ious is not None:
        best_idx = int(torch.argmax(ious[0]).item())
        logits = logits[:, best_idx:best_idx+1]
    # 官方做法不对低分辨率mask做平滑插值：先在原尺度二值化，再用NEAREST上采，避免灰度扩散
    probs_256 = torch.sigmoid(logits)[0, 0]           # (256,256)
    mask_256 = (probs_256 > float(args.threshold)).float()
    mask = F.interpolate(mask_256[None, None], size=(1024, 1024), mode='nearest')[0, 0]

    # 如果掩码全空，回退输出热力图方便排查
    save_heatmap_also = False
    if mask.sum() == 0:
        save_heatmap_also = True

    # 叠加可视化：原图+掩码+点高亮
    base = Image.open(args.image).convert('RGB').resize((1024, 1024))
    overlay = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
    # 掩码半透明绿色（严格二值化透明度）
    mask_rgba = Image.fromarray((mask.cpu().numpy() * 255).astype('uint8')).convert('L')
    green = Image.new('RGBA', (1024, 1024), (0, 255, 0, 120))
    overlay.paste(green, (0, 0), mask_rgba)
    # 点高亮（红色小圆）
    draw = ImageDraw.Draw(overlay)
    r = 6
    cx, cy = point_xy
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0, 255))
    vis = Image.alpha_composite(base.convert('RGBA'), overlay)
    vis.convert('RGB').save(args.out)
    print(f"Saved visualization to {args.out}; point=({point_xy[0]:.1f},{point_xy[1]:.1f}), threshold={args.threshold}, multimask={max(1,int(args.multimask))}")
    if save_heatmap_also:
        import numpy as np
        # 归一化热力图
        # 可视化仍基于原生(256x256)概率，避免平滑造成误导
        hm = (probs_256 - probs_256.min()) / (probs_256.max() - probs_256.min() + 1e-6)
        hm_img = Image.fromarray((hm.cpu().numpy() * 255).astype('uint8')).resize((1024,1024))
        hm_img = hm_img.convert('L')
        # 叠加黄色热力
        heat = Image.new('RGBA', (1024, 1024), (255, 200, 0, 140))
        ov2 = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
        ov2.paste(heat, (0, 0), hm_img)
        vis2 = Image.alpha_composite(base.convert('RGBA'), ov2)
        out_hm = os.path.splitext(args.out)[0] + '_heatmap.png'
        vis2.convert('RGB').save(out_hm)
        print(f"Mask empty -> saved heatmap to {out_hm}")


if __name__ == '__main__':
    main()



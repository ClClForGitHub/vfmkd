import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

# 路径注入，使用本地vendor sam2
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SAM2_PARENT = _ROOT / 'vfmkd' / 'sam2'
if str(_SAM2_PARENT) not in sys.path:
    sys.path.insert(0, str(_SAM2_PARENT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_image_rgb(path: str, size: int = 1024) -> Image.Image:
    return Image.open(path).convert('RGB').resize((size, size))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--point', type=float, nargs=2, default=[512, 512])
    ap.add_argument('--label', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out', type=str, default='outputs/sam2_ref_mask.png')
    ap.add_argument('--checkpoint', type=str, default='weights/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--config', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    args = ap.parse_args()

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    model = build_sam2(config_file=args.config, ckpt_path=args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    img = load_image_rgb(args.image, size=1024)
    predictor.set_image(img)

    # SAM2 接口期望 (N,2) 像素坐标和 (N,) 标签
    import numpy as np
    pts = np.array([args.point], dtype=np.float32)
    lbl = np.array([args.label], dtype=np.int32)

    masks, ious, lowres = predictor.predict(
        point_coords=pts[None, ...],
        point_labels=lbl[None, ...],
        multimask_output=True,
        return_logits=False,
    )
    # 诊断信息
    import numpy as np
    print(f"masks.shape={masks.shape}, ious.shape={ious.shape}, lowres.shape={lowres.shape}")
    print(f"ious[0]={ious[0]}")

    # 保存三张候选掩码以便检查
    best = int(np.argmax(ious[0]))
    mask = masks[0, best]
    if mask.dtype != np.uint8:
        mask_vis = (mask * 255).astype('uint8')
    else:
        mask_vis = mask

    # 保存可视化：原图+掩码+点
    base = img.copy()
    overlay = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
    mask_img = Image.fromarray(mask_vis).convert('L').resize((1024, 1024), Image.NEAREST)
    green = Image.new('RGBA', (1024, 1024), (0, 255, 0, 120))
    overlay.paste(green, (0, 0), mask_img)
    draw = ImageDraw.Draw(overlay)
    r = 6
    cx, cy = args.point
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0, 255))
    vis = Image.alpha_composite(base.convert('RGBA'), overlay)
    Path(os.path.dirname(args.out) or '.').mkdir(parents=True, exist_ok=True)
    vis.convert('RGB').save(args.out)
    print(f"Saved SAM2 reference to {args.out}")

    # 另外保存所有候选掩码与像素统计
    out_stem = Path(args.out).with_suffix('')
    for i in range(masks.shape[0]):
        mi = masks[i].astype('uint8') * 255
        Image.fromarray(mi).save(f"{out_stem}_cand{i}.png")
        print(f"mask[{i}] sum={(mi>0).sum()} pixels, iou={ious[i]:.4f}")

    # 保存低分辨率掩码候选可视
    for i in range(lowres.shape[0]):
        lr = lowres[i]
        lr = (lr - lr.min()) / (lr.max() - lr.min() + 1e-6)
        Image.fromarray((lr * 255).astype('uint8')).resize((1024, 1024), Image.BILINEAR).save(f"{out_stem}_lowres{i}.png")


if __name__ == '__main__':
    main()



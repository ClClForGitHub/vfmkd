import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import sys
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_adapter_head import SAM2SegAdapterHead


def load_image_rgb(path: str, size: int = 1024) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size))
    x = TF.to_tensor(img).unsqueeze(0)
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--yolov8_weight', type=str, default='')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_stem', type=str, default='outputs/adapter_cand')
    ap.add_argument('--point', type=float, nargs=2, default=[512, 512])
    ap.add_argument('--label', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    backbone = YOLOv8Backbone({
        'model_size': 'n',
        'pretrained': False,
        'external_weight_path': args.yolov8_weight if args.yolov8_weight else None,
    }).to(device).eval()

    head = SAM2SegAdapterHead({
        'image_size': 1024,
        'hidden_dim': 256,
        'in_channels': backbone.get_feature_dims(),
        'backbone_strides': backbone.get_feature_strides(),
        'use_high_res_features': False,
        'num_multimask_outputs': 3,
    }).to(device).eval()

    x = load_image_rgb(args.image, 1024).to(device)
    feats = backbone(x)

    pts = torch.tensor([[args.point]], device=device, dtype=torch.float32)
    lbl = torch.tensor([[args.label]], device=device, dtype=torch.int32)

    out = head(feats, point_coords=pts, point_labels=lbl, num_multimask_outputs=3)
    logits = out['low_res_logits']  # [1,3,256,256]

    # 逐cand生成二值掩码：先在256上阈值，再NEAREST上采到1024，保存纯二值灰度
    Path(os.path.dirname(args.out_stem) or '.').mkdir(parents=True, exist_ok=True)
    for i in range(logits.shape[1]):
        p = torch.sigmoid(logits[0, i])
        m = (p > float(args.threshold)).float()
        m_up = F.interpolate(m[None, None], size=(1024, 1024), mode='nearest')[0, 0]
        Image.fromarray((m_up.cpu().numpy() * 255).astype('uint8')).save(f"{args.out_stem}_cand{i}.png")
        print(f"cand{i}: sum={(m_up>0.5).sum().item()} pixels")


if __name__ == '__main__':
    main()



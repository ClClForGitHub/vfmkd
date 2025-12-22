import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ensure package path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vfmkd.teachers.sam2_teacher import SAM2Teacher
from vfmkd.models.backbones.repvit_backbone import RepViT, RepViTBackbone
from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter


def load_image(path: str, size: int = 1024, device: str = 'cuda') -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size))
    arr = np.array(img, copy=True)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0).to(device)


def save_heatmap(t: torch.Tensor, out_path: Path) -> None:
    t = t.detach().float().cpu()
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    img = (t * 255).clamp(0, 255).byte().numpy()
    Image.fromarray(img).save(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--teacher_model', type=str, default='sam2.1_hiera_b+')
    ap.add_argument('--checkpoint', type=str, default='weights/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--repvit_variant', type=str, default='m1')
    ap.add_argument('--weight', type=str, default='outputs/repvit_align_cosin_e50.pth')
    ap.add_argument('--out_dir', type=str, default='outputs/compare_emb')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Teacher (SAM2) for native image embeddings
    teacher = SAM2Teacher({
        'model': args.teacher_model,
        'checkpoint': args.checkpoint,
        'device': args.device,
    })
    teacher.model.eval()

    # Student (RepViT + align adapter)
    repvit = RepViTBackbone({'arch': args.repvit_variant, 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None})
    assert isinstance(repvit, torch.nn.Module), 'RepViTBackbone build failed'
    repvit = repvit.to(device).eval()
    adapter = RepViTAlignAdapter().to(device).eval()
    if args.weight and Path(args.weight).exists():
        ckpt = torch.load(args.weight, map_location='cpu')
        if 'backbone' in ckpt and hasattr(repvit, 'load_state_dict') and ckpt['backbone'] is not None:
            try:
                repvit.load_state_dict(ckpt['backbone'], strict=False)
            except Exception as e:
                print(f"[WARN] load repvit backbone failed: {e}")
        if 'adapter' in ckpt and ckpt['adapter'] is not None:
            try:
                adapter.load_state_dict(ckpt['adapter'], strict=False)
            except Exception as e:
                print(f"[WARN] load adapter failed: {e}")

    # Load image
    img_path = Path(args.image)
    x = load_image(str(img_path), size=1024, device=device)

    # Teacher features
    with torch.no_grad():
        np_img = np.array(Image.open(str(img_path)).convert('RGB'))
        feats_t = teacher.extract_features(np_img, image_ids=['cmp'], save_features=False)
        # 约定键名：使用 s16 主分辨率
        t = None
        # 优先常用键
        prefer_keys = ['image_embeddings_s16', 'image_emb_s16', 'image_embeddings']
        for k in prefer_keys:
            if k in feats_t and isinstance(feats_t[k], torch.Tensor):
                t = feats_t[k]
                break
        if t is None:
            # 回退：寻找形状为 (B,256,64,64) 的张量
            for k, v in feats_t.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    if v.shape[1] == 256 and v.shape[-2] in (64, 65) and v.shape[-1] in (64, 65):
                        t = v
                        break
        assert t is not None, f'SAM2 teacher image embedding (s16) not found. keys={list(feats_t.keys())}'
        t = t.to(device).float()  # (1,256,64,64)

    # Student features
    with torch.no_grad():
        feats_s = repvit(x)
        s = adapter(feats_s).float()  # (1,256,64,64)

    # Resize to same size if mismatch
    if s.shape[-2:] != t.shape[-2:]:
        s = F.interpolate(s, size=t.shape[-2:], mode='bilinear', align_corners=False)

    # Metrics
    cos = F.cosine_similarity(F.normalize(s, dim=1), F.normalize(t, dim=1), dim=1)[0]  # (64,64)
    l2 = ((s - t) ** 2).mean(dim=1)[0]  # (64,64)

    stats = {
        'cos_mean': float(cos.mean().item()),
        'cos_min': float(cos.min().item()),
        'cos_max': float(cos.max().item()),
        'l2_mean': float(l2.mean().item()),
        'l2_min': float(l2.min().item()),
        'l2_max': float(l2.max().item()),
    }
    (out_dir / 'stats.txt').write_text('\n'.join([f'{k}: {v}' for k, v in stats.items()]), encoding='utf-8')

    save_heatmap(cos, out_dir / 'cosine_64x64.png')
    save_heatmap(l2, out_dir / 'l2_64x64.png')

    print('[COMPARE] ', stats)
    print(f"Saved heatmaps & stats to: {out_dir}")


if __name__ == '__main__':
    main()



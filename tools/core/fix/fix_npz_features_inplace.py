#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import time, os
import tempfile
import json
import cv2
from typing import List, Optional


def parse_args():
    ap = argparse.ArgumentParser(description='原地修复NPZ特征（重算P4/P5）')
    ap.add_argument('--samples', type=str, required=True, help='样本清单，每行一个jpg绝对路径')
    ap.add_argument('--weights', type=str, default='weights/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--npz-dirs', type=str, default='', help='集中NPZ目录，逗号分隔；将递归搜索 <stem>_features.npz 或 <stem>_sam2_features.npz')
    ap.add_argument('--log', type=str, required=True)
    ap.add_argument('--max-images', type=int, default=0, help='0为不限')
    return ap.parse_args()


def load_sam2(weights_path: Path, device: torch.device):
    root = Path(__file__).parent.parent
    sam2_pkg = root / 'vfmkd' / 'sam2'
    if str(sam2_pkg) not in sys.path:
        sys.path.insert(0, str(sam2_pkg))
    from sam2.build_sam import build_sam2
    from sam2.utils.transforms import SAM2Transforms
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    cfg_name = 'sam2.1/sam2.1_hiera_b+.yaml'
    config_dir = sam2_pkg / 'sam2' / 'configs'
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        model = build_sam2(config_file=cfg_name, ckpt_path=str(weights_path), device=str(device))
    model.eval()
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
    return model, transforms


def forward_image(model, transforms, image_rgb, device):
    img_t = transforms(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.image_encoder(img_t)
    return out


def atomic_replace(target_path: Path, write_func):
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    # 在目标目录创建临时文件，避免跨设备问题
    import tempfile
    fd, tmp_name = tempfile.mkstemp(suffix='.npz', dir=str(target_dir))
    os.close(fd)
    try:
        write_func(tmp_name)
        os.replace(tmp_name, target_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


def find_npz_for_stem(stem: str, img_dir: Path, search_dirs: List[Path]) -> Optional[Path]:
    # 优先同目录 _features
    cand1 = img_dir / f'{stem}_features.npz'
    if cand1.exists():
        return cand1
    # 其次同目录 _sam2_features
    cand2 = img_dir / f'{stem}_sam2_features.npz'
    if cand2.exists():
        return cand2
    # 在集中目录递归搜索
    patterns = [f'{stem}_features.npz', f'{stem}_sam2_features.npz']
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            hits = list(d.rglob(pat))
            if hits:
                return hits[0]
    return None


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Path(args.weights)
    model, tf = load_sam2(weights, device)

    search_dirs = [Path(p.strip()) for p in args.npz_dirs.split(',') if p.strip()]

    lines = [p for p in Path(args.samples).read_text().splitlines() if p.strip()]
    if args.max_images and args.max_images > 0:
        lines = lines[:args.max_images]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.time()

    with open(log_path, 'w', encoding='utf-8') as lf:
        for idx, jpg_str in enumerate(lines, 1):
            jpg = Path(jpg_str)
            stem = jpg.stem
            npz_path = find_npz_for_stem(stem, jpg.parent, search_dirs)
            if npz_path is None or not npz_path.exists():
                lf.write(f'[SKIP] NPZ not found for {jpg}\n'); lf.flush(); continue
            img_bgr = cv2.imread(str(jpg))
            if img_bgr is None:
                lf.write(f'[SKIP] Image not readable {jpg}\n'); lf.flush(); continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            out = forward_image(model, tf, img_rgb, device)
            p4 = out['backbone_fpn'][2].detach().cpu().numpy()
            p5 = out['backbone_fpn'][3].detach().cpu().numpy()

            # 写临时文件并原子替换
            with np.load(npz_path, allow_pickle=True) as old:
                data = {k: old[k] for k in old.files}
            data['P4_S16'] = p4
            data['IMAGE_EMB_S16'] = p4  # 兼容键
            data['P5_S32'] = p5
            def _write(dst):
                np.savez(dst, **data)
            atomic_replace(npz_path, _write)
            dt = time.time() - t0
            lf.write(f'[OK] {idx}/{len(lines)} {jpg.name} -> {npz_path} dt={dt:.3f}s\n'); lf.flush()

    dur = time.time() - t0_all
    (log_path.parent/'metrics.txt').write_text(f'duration_sec={int(dur)}\n')
    print(f'[DONE] inplace fix completed in {dur:.1f}s')


if __name__ == '__main__':
    main()

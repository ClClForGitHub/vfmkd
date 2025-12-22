#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import time
import os
import cv2
from typing import List, Optional


def parse_args():
    ap = argparse.ArgumentParser(description='批量替换NPZ中的P4/P5特征，并标记feature_flag=1')
    ap.add_argument('--root', type=str, required=True, help='递归扫描的根目录（如 /home/team/zouzhiyuan/dataset/sa1b）')
    ap.add_argument('--weights', type=str, default='weights/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--log', type=str, required=True)
    ap.add_argument('--max-files', type=int, default=0, help='限制处理文件数，0为不限')
    return ap.parse_args()


def load_sam2(weights_path: Path, device: torch.device):
    root_dir = Path(__file__).parent.parent
    sam2_pkg = root_dir / 'vfmkd' / 'sam2'
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


def atomic_write_npz(target_path: Path, data_dict: dict):
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp_name = tempfile.mkstemp(suffix='.npz', dir=str(target_dir))
    os.close(fd)
    try:
        np.savez(tmp_name, **data_dict)
        os.replace(tmp_name, target_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Path(args.weights)
    model, tf = load_sam2(weights, device)

    root = Path(args.root)
    # 收集NPZ（两种命名），按文件名排序，保证仅处理一次
    candidates: List[Path] = []
    for pat in ('*_features.npz', '*_sam2_features.npz'):
        candidates.extend(sorted(root.rglob(pat)))

    # 去重：优先 *_features.npz
    seen = set()
    unique_npz: List[Path] = []
    for p in candidates:
        stem = p.name
        key = stem.replace('_sam2_features.npz', '').replace('_features.npz', '')
        if key in seen:
            continue
        seen.add(key)
        unique_npz.append(p)

    if args.max_files and args.max_files > 0:
        unique_npz = unique_npz[:args.max_files]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    with open(log_path, 'w', encoding='utf-8') as lf:
        for idx, npz_path in enumerate(unique_npz, 1):
            stem = npz_path.stem.replace('_features','').replace('_sam2_features','')
            img_path = root / f'{stem}.jpg'
            if not img_path.exists():
                lf.write(f'[SKIP] image not found for {npz_path}\n'); lf.flush(); skipped += 1; continue

            try:
                with np.load(npz_path, allow_pickle=True) as old:
                    data = {k: old[k] for k in old.files}
            except Exception as e:
                lf.write(f'[SKIP] failed to read NPZ {npz_path}: {e}\n'); lf.flush(); skipped += 1; continue

            if str(data.get('feature_flag', 0)) == '1' or (isinstance(data.get('feature_flag', 0), np.ndarray) and data.get('feature_flag', np.array(0)).item() == 1):
                lf.write(f'[SKIP] already processed {npz_path}\n'); lf.flush(); skipped += 1; continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                lf.write(f'[SKIP] unreadable image {img_path}\n'); lf.flush(); skipped += 1; continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            out = forward_image(model, tf, img_rgb, device)
            p4 = out['backbone_fpn'][2].detach().cpu().numpy()
            p5 = out['backbone_fpn'][3].detach().cpu().numpy()

            # 仅替换这两个键，其他不变
            data['P4_S16'] = p4
            data['P5_S32'] = p5
            data['feature_flag'] = 1

            try:
                atomic_write_npz(npz_path, data)
            except Exception as e:
                lf.write(f'[ERR] write failed {npz_path}: {e}\n'); lf.flush(); continue

            dt = time.time() - t0
            processed += 1
            lf.write(f'[OK] {idx}/{len(unique_npz)} {npz_path} dt={dt:.3f}s\n'); lf.flush()

    print(f'[DONE] processed={processed} skipped={skipped} total={len(unique_npz)}')


if __name__ == '__main__':
    main()



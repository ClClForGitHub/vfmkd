from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from pycocotools import mask as mask_utils
import cv2


def bbox_from_json(json_path: Path, target_size: int = 1024, pad_ratio: float = 0.05) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """从 JSON 里选一个实例的最小外接框，映射到 target_size×target_size。
    策略：优先使用 RLE 的 toBbox；无则 decode 取最大连通域外接框。
    返回：
      - boxes_tensor: (1,4) 的 float32 tensor，坐标已映射到 [0,target_size-1]
      - box_xyxy_int: (x0,y0,x1,y1) int 元组（同一坐标系）
    """
    data = None
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
    anns = data.get('annotations', [])
    W = int(data['image']['width'])
    H = int(data['image']['height'])
    if not anns:
        # 无标注，则用中心较小框
        x0 = int(0.25 * target_size)
        y0 = int(0.25 * target_size)
        x1 = int(0.75 * target_size)
        y1 = int(0.75 * target_size)
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
        return box, (x0, y0, x1, y1)

    # 先选一个“较合理”的实例（面积最接近 10% 作为启发式）
    img_area = float(W * H)
    best = None
    TARGET = 0.10
    for ann in anns:
        try:
            a = float(mask_utils.area(ann['segmentation']))
            r = a / img_area
            score = abs(r - TARGET)
            if (best is None) or (score < best[0]):
                best = (score, ann['segmentation'])
        except Exception:
            continue
    rle = best[1] if best else anns[0]['segmentation']

    # 优先 toBbox，失败再 decode
    try:
        x, y, w, h = mask_utils.toBbox(rle)
        xs, ys, ws, hs = float(x), float(y), float(w), float(h)
    except Exception:
        m = mask_utils.decode(rle).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = 1 + int(np.argmax(areas))
            xs, ys, ws, hs = [float(stats[idx, k]) for k in [0, 1, 2, 3]]
        else:
            ys_arr, xs_arr = np.where(m > 0)
            ys, xs = float(ys_arr.min()), float(xs_arr.min())
            hs = float(ys_arr.max() - ys_arr.min() + 1)
            ws = float(xs_arr.max() - xs_arr.min() + 1)

    # 映射到 target_size
    sx = target_size / float(W)
    sy = target_size / float(H)
    x0 = int(round(xs * sx))
    y0 = int(round(ys * sy))
    x1 = int(round((xs + ws) * sx))
    y1 = int(round((ys + hs) * sy))
    # padding
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(target_size - 1, x1 + px)
    y1 = min(target_size - 1, y1 + py)

    box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
    return box, (x0, y0, x1, y1)



from pathlib import Path
from typing import Tuple, List, Optional
import json

import numpy as np
import torch
from pycocotools import mask as mask_utils
import cv2


def score_instance_by_bbox(
    rle, 
    img_w: int, 
    img_h: int, 
    img_area: float,
    target_ratio: float = 0.10,
    min_ratio: float = 0.03,
    max_ratio: float = 0.30,
) -> Tuple[float, Optional[List[float]], float, float, float]:
    """
    使用RLE的面积与bbox计算打分，返回(score, bbox[x,y,w,h], ratio, fill, aspect).
    
    策略：
    - 越接近target_ratio越好
    - 惩罚极端长宽比与条带（横向/纵向跨满）
    - 偏好较高填充度 fill = area/(w*h)
    不解码整mask，速度更快。
    
    Args:
        rle: RLE编码的mask
        img_w: 图像宽度
        img_h: 图像高度
        img_area: 图像总面积
        target_ratio: 目标实例面积占比，默认0.10
        min_ratio: 最小面积占比阈值，默认0.03
        max_ratio: 最大面积占比阈值，默认0.30
    
    Returns:
        (score, bbox, ratio, fill, aspect):
        - score: 打分（越小越好）
        - bbox: [x, y, w, h] 或 None
        - ratio: 面积占比
        - fill: 填充度（area/(w*h)）
        - aspect: 长宽比（max(w/h, h/w)）
    """
    try:
        area = float(mask_utils.area(rle))
        if area <= 0:
            return (float('inf'), None, 0.0, 0.0, 0.0)
        bbox = mask_utils.toBbox(rle)  # [x, y, w, h]
        x, y, w, h = [float(v) for v in bbox]
        if w <= 1 or h <= 1:
            return (float('inf'), None, 0.0, 0.0, 0.0)
        ratio = area / img_area
        aspect = max(w / h, h / w)
        width_ratio = w / float(img_w)
        height_ratio = h / float(img_h)
        fill = area / (w * h)

        size_penalty = abs(ratio - target_ratio)
        if ratio < min_ratio or ratio > max_ratio:
            size_penalty += 0.5
        compact_penalty = max(0.0, (aspect - 3.0) / 3.0)  # >3 开始惩罚
        stripe_penalty = 0.0
        if width_ratio > 0.8 and height_ratio < 0.25:
            stripe_penalty += 1.0
        if height_ratio > 0.8 and width_ratio < 0.25:
            stripe_penalty += 1.0
        sparsity_penalty = max(0.0, 0.2 - fill)

        score = size_penalty + compact_penalty + stripe_penalty + sparsity_penalty
        return (score, [x, y, w, h], ratio, fill, aspect)
    except Exception:
        return (float('inf'), None, 0.0, 0.0, 0.0)


def bbox_from_json(
    json_path: Path, 
    target_size: int = 1024, 
    pad_ratio: float = 0.05,
    strategy: str = 'best_single',
    num_boxes: int = 1,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    从 JSON 里选一个或多个实例的最小外接框，映射到 target_size×target_size。
    
    Args:
        json_path: JSON标注文件路径
        target_size: 目标尺寸，默认1024
        pad_ratio: 框的padding比例，默认0.05
        strategy: 选择策略
            - 'best_single': 选择最合适的单个实例（默认）
            - 'first_n': 选择前N个实例（按score排序）
            - 'all': 选择所有实例
        num_boxes: 当strategy='first_n'时，选择的框数量
    
    Returns:
        - boxes_tensor: (N,4) 的 float32 tensor，坐标已映射到 [0,target_size-1]
        - box_xyxy_int: (x0,y0,x1,y1) int 元组（第一个框的坐标）
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    anns = data.get('annotations', [])
    W = int(data['image']['width'])
    H = int(data['image']['height'])
    img_area = float(W * H)
    
    if not anns:
        # 无标注，则用中心较小框
        x0 = int(0.25 * target_size)
        y0 = int(0.25 * target_size)
        x1 = int(0.75 * target_size)
        y1 = int(0.75 * target_size)
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
        return box, (x0, y0, x1, y1)

    # 对每个实例打分
    scored_anns = []
    for ann in anns:
        try:
            score, bbox, ratio, fill, aspect = score_instance_by_bbox(
                ann['segmentation'], W, H, img_area
            )
            if bbox is not None:
                scored_anns.append((score, bbox, ratio, ann['segmentation']))
        except Exception:
            continue
    
    if not scored_anns:
        # 没有有效实例，使用中心框
        x0 = int(0.25 * target_size)
        y0 = int(0.25 * target_size)
        x1 = int(0.75 * target_size)
        y1 = int(0.75 * target_size)
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
        return box, (x0, y0, x1, y1)
    
    # 按策略选择实例
    if strategy == 'best_single':
        selected = [min(scored_anns, key=lambda x: x[0])]
    elif strategy == 'first_n':
        scored_anns.sort(key=lambda x: x[0])  # 按score排序
        selected = scored_anns[:num_boxes]
    elif strategy == 'all':
        selected = scored_anns
        else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 提取框并映射到target_size
    boxes_list = []
    sx = target_size / float(W)
    sy = target_size / float(H)
    
    for score, bbox, ratio, rle in selected:
        x, y, w, h = bbox
        
        # 映射到target_size
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = int(round((x + w) * sx))
        y1 = int(round((y + h) * sy))
        
    # padding
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(target_size - 1, x1 + px)
    y1 = min(target_size - 1, y1 + py)

        boxes_list.append([x0, y0, x1, y1])

    # 转换为tensor
    boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)

    # 返回第一个框的坐标（用于兼容）
    first_box = boxes_list[0]
    return boxes_tensor, tuple(first_box)

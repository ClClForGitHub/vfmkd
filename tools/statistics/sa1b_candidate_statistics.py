#!/usr/bin/env python3
"""
SA-1B 候选体/合体统计脚本
---------------------------------

根据用户提供的口径规范，针对 SA-1B JSON 标注与 NPZ 边缘特征，生成以下三张 CSV：

1. pairs_stats.csv
2. cluster_gain.csv
3. candidate_stats.csv

核心流程：
  - 随机采样 10k 图像（可配置数量/随机种子）
  - 按“硬筛”规则过滤合格掩码
  - 构建碎片池，计算片对亲和度 A_ij 以及桥接/缝隙等指标
  - 受约束凝聚：按 A_ij 从大到小尝试合并碎片，输出基础分增益
  - 对所有候选体（单片/合体）计算 S_base 与 S_total
  - 导出 CSV，并可选输出分位统计

说明：
  - 该脚本以“口径准确”为第一优先级，实现尽可能接近日常训练代码的逻辑
  - 性能优化（批处理、缓存、并行）留作后续迭代；当前版本偏重可读性与正确性
  - 使用 numpy / scipy / OpenCV / pycocotools

Author: GPT-5 Codex (Cursor Agent)
Date: 2025-11-09
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_fill_holes, distance_transform_edt


# -----------------------------------------------------------------------------
# 常量 & 辅助函数
# -----------------------------------------------------------------------------

EPS = 1e-6
EDGE_CONST_FALLBACK_STD = 1e-4
S_EDGE_IQR_MIN = 1e-3
LINE_THICKNESS = 3
MAX_NEIGHBORS = 12
THETA_DEFAULT = 0.60
TAU_DEFAULT = 0.03
R_CLOSE = 2


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def gaussian_log_norm(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    return np.exp(-((np.log(x + EPS) - math.log(mu)) ** 2) / (2.0 * sigma**2))


def ensure_bool(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(bool)


def make_disk(radius: int) -> np.ndarray:
    """生成 OpenCV 可用的圆盘结构元素"""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def clamp01(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, np.ndarray):
        return np.clip(x, 0.0, 1.0)
    return max(0.0, min(1.0, x))


def safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    count = mask.sum()
    if count <= 0:
        return 0.0
    return float(values[mask].mean())


def compute_signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    计算掩码的 Signed Distance Field (SDF)，掩码内为正，外为负。
    """
    mask_bool = ensure_bool(mask)
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool, dtype=np.float32)
    inside = distance_transform_edt(mask_bool)
    outside = distance_transform_edt(~mask_bool)
    sdf = inside - outside
    return sdf


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def bbox_to_xyxy(bbox_xywh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def compute_convex_hull_area(mask: np.ndarray) -> float:
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    area = cv2.contourArea(hull)
    return float(area)


def compute_hole_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    filled = binary_fill_holes(mask)
    area = mask.sum()
    filled_area = filled.sum()
    if filled_area == 0:
        return 0.0
    return float((filled_area - area) / max(filled_area, 1))


def touch_edge_ratios(mask: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape
    top_len = mask[0, :].sum() / max(w, 1)
    bottom_len = mask[-1, :].sum() / max(w, 1)
    left_len = mask[:, 0].sum() / max(h, 1)
    right_len = mask[:, -1].sum() / max(h, 1)
    return {
        "top": float(top_len),
        "bottom": float(bottom_len),
        "left": float(left_len),
        "right": float(right_len),
    }


def upsample_edge_map(edge_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """将 edge_{k} 上采样到原图尺寸"""
    th, tw = target_hw
    if edge_map.shape == (th, tw):
        return edge_map.astype(np.float32)
    return cv2.resize(edge_map.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)


def quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def bbox_iou_xywh(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area + EPS
    return inter_area / union


def compute_center_distance(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    cx_a = ax + 0.5 * aw
    cy_a = ay + 0.5 * ah
    cx_b = bx + 0.5 * bw
    cy_b = by + 0.5 * bh
    return float(math.hypot(cx_a - cx_b, cy_a - cy_b))


def ensure_npz_edge(npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ["edge_256x256", "edge_512x512", "edge"]:
        if key in npz_data.files:
            arr = npz_data[key]
            if arr.ndim == 2:
                return arr.astype(np.float32)
    raise KeyError("NPZ 中缺少 edge_* 边缘图")


# -----------------------------------------------------------------------------
# 评分与统计函数
# -----------------------------------------------------------------------------


def compute_edge_regions(mask: np.ndarray, sdf: np.ndarray, edge_map: np.ndarray, te: float) -> Dict[str, float]:
    rbd = np.abs(sdf) <= 3.0
    rout = np.logical_and(sdf >= -7.0, sdf < -3.0)
    rin = np.logical_and(sdf > 3.0, sdf <= 7.0)
    if edge_map.std() < EDGE_CONST_FALLBACK_STD:
        return {
            "E_bd": 0.5,
            "E_in": 0.5,
            "E_out": 0.5,
            "ER": 0.5,
            "d_in": 0.5,
            "d_near": 0.5,
            "delta_e": 0.0,
            "r_tex": 1.0,
        }

    E_bd = safe_mean(rbd, edge_map)
    E_in = safe_mean(rin, edge_map)
    E_out = safe_mean(rout, edge_map)
    indicator = edge_map >= te
    ER = safe_mean(rbd, indicator)
    d_in = mask.sum()
    d_in_val = 0.0 if d_in == 0 else float(indicator[mask].sum() / max(d_in, 1))
    rout_count = rout.sum()
    d_near_val = 0.0 if rout_count == 0 else float(indicator[rout].sum() / max(rout_count, 1))
    r_tex = d_in_val / (d_near_val + 1e-6)

    return {
        "E_bd": float(E_bd),
        "E_in": float(E_in),
        "E_out": float(E_out),
        "ER": float(ER),
        "d_in": float(d_in_val),
        "d_near": float(d_near_val),
        "delta_e": float(E_bd - 0.5 * (E_in + E_out)),
        "r_tex": float(r_tex),
    }


def compute_candidate_scores_single(
    inst: InstanceMetrics,
    delta_e_array: np.ndarray,
    d_in_array: np.ndarray,
    iqr_min: float = S_EDGE_IQR_MIN,
) -> Dict[str, float]:
    delta_e_vals = delta_e_array
    iqr = np.quantile(delta_e_vals, 0.75) - np.quantile(delta_e_vals, 0.25) if delta_e_vals.size > 0 else 0.0
    s_iqr = iqr / 1.349 + 1e-6
    if s_iqr < iqr_min:
        s_edge = 0.5
    else:
        s_edge = float(sigmoid(inst.delta_e / s_iqr))

    q_json = math.sqrt(max(inst.predicted_iou, 0.0) * max(inst.stability, 0.0))

    s_size = float(gaussian_log_norm(inst.area_ratio, mu=0.015, sigma=0.6))

    s_ctr = 0.5 * math.exp(-((inst.cx - 0.5) ** 2) / (2 * 0.25**2)) + 0.5 * math.exp(-((inst.cy - 0.6) ** 2) / (2 * 0.20**2))

    s_vis = float(clamp01((inst.v_visible - 0.6) / 0.4))

    s_tex = float(np.exp(-((math.log(inst.r_tex + EPS) - math.log(1.2)) ** 2) / (2 * 0.35**2)))

    aspect_term = clamp01(1 - abs(math.log(inst.aspect + EPS)) / math.log(2.5))
    solidity_term = clamp01((inst.solidity - 0.75) / 0.25)
    holes_term = clamp01(1 - inst.hole_frac / 0.15)
    s_shape = float(aspect_term * solidity_term * holes_term)

    d_in_vals = d_in_array
    if d_in_vals.size == 0:
        d_in_p10 = 0.0
    else:
        d_in_p10 = float(np.quantile(d_in_vals, 0.10))

    p_stuff = 1.0
    top_touch = inst.touch_edges.get("top", 0.0)
    bottom_touch = inst.touch_edges.get("bottom", 0.0)
    W_contact = max(top_touch, bottom_touch)
    area_ratio = inst.area_ratio
    r = inst.aspect

    if area_ratio > 0.20 and W_contact > 0.5:
        p_stuff = min(p_stuff, 0.1)
    if r > 6.0 and inst.cy > 0.8:
        p_stuff = min(p_stuff, 0.3)
    if inst.solidity > 0.95 and inst.d_in < d_in_p10:
        p_stuff = min(p_stuff, 0.3)
    if (r > 8.0 or r < 0.125) and area_ratio < 0.02:
        p_stuff = min(p_stuff, 0.4)

    s_base = (
        q_json**0.20
        * s_size**0.15
        * s_ctr**0.10
        * s_vis**0.20
        * s_edge**0.15
        * s_tex**0.08
        * s_shape**0.07
    )
    s_total = s_base * (p_stuff**0.05)

    return {
        "q_json": q_json,
        "s_size": s_size,
        "s_ctr": s_ctr,
        "s_vis": s_vis,
        "s_edge": s_edge,
        "s_tex": s_tex,
        "s_shape": s_shape,
        "p_stuff": p_stuff,
        "s_base": s_base,
        "s_total": s_total,
    }


# -----------------------------------------------------------------------------
# Image-level处理
# -----------------------------------------------------------------------------


def load_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_edge_for_image(npz_dir: Path, stem: str, hw: Tuple[int, int]) -> np.ndarray:
    npz_path = npz_dir / f"{stem}_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"缺少 NPZ 文件: {npz_path}")
    with np.load(npz_path) as data:
        edge = ensure_npz_edge(data)
    edge_up = upsample_edge_map(edge, hw)
    if edge_up.max() > 1.0:
        edge_up = np.clip(edge_up, 0.0, 1.0)
    return edge_up


def decode_mask(ann: Dict, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if seg is None:
        return np.zeros((height, width), dtype=bool)
    mask = mask_utils.decode(seg)
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def compute_instances(
    data: Dict,
    edge_map: np.ndarray,
    image_id: str,
) -> List[InstanceMetrics]:
    height = int(data["image"]["height"])
    width = int(data["image"]["width"])
    annotations: List[Dict] = data.get("annotations", [])
    te = quantile(edge_map, 0.85)
    instances: List[InstanceMetrics] = []

    masks_cache: List[np.ndarray] = []
    bbox_list: List[Tuple[float, float, float, float]] = []
    ann_ids: List[str] = []
    pred_list: List[float] = []
    stab_list: List[float] = []

    # First pass: decode masks & compute metrics except visibility
    for ann in annotations:
        ann_id = ann.get("id", ann.get("ann_id", len(instances)))
        bbox = mask_utils.toBbox(ann["segmentation"]).astype(float)
        x, y, w, h = bbox.tolist()
        if w <= 0 or h <= 0:
            continue
        area_json = float(ann.get("area", w * h))
        area_ratio_json = area_json / (height * width)
        mask = decode_mask(ann, height, width)
        area_px = int(mask.sum())
        if area_px <= 0:
            continue
        area_ratio = area_px / (height * width)
        cx = (x + 0.5 * w) / width
        cy = (y + 0.5 * h) / height
        aspect = max(w, h) / max(min(w, h), EPS)

        solidity = 0.0
        hull_area = compute_convex_hull_area(mask)
        if hull_area > 0:
            solidity = float(area_px / hull_area)

        hole_frac = compute_hole_fraction(mask)
        touch_edges = touch_edge_ratios(mask)
        edge_stats = compute_edge_regions(mask, compute_signed_distance(mask), edge_map, te)

        inst = InstanceMetrics(
            image_id=image_id,
            ann_id=str(ann_id),
            mask=mask,
            area_px=area_px,
            bbox_xywh=(float(x), float(y), float(w), float(h)),
            cx=float(cx),
            cy=float(cy),
            aspect=float(aspect),
            solidity=float(solidity),
            hole_frac=float(hole_frac),
            touch_edges=touch_edges,
            predicted_iou=float(ann.get("predicted_iou", 0.0)),
            stability=float(ann.get("stability_score", 0.0)),
            v_visible=0.0,  # placeholder
            delta_e=edge_stats["delta_e"],
            er=edge_stats["ER"],
            er_in=edge_stats["E_in"],
            er_out=edge_stats["E_out"],
            d_in=edge_stats["d_in"],
            d_near=edge_stats["d_near"],
            r_tex=edge_stats["r_tex"],
            area_ratio=float(area_ratio),
            edge_quantile=te,
        )
        instances.append(inst)
        masks_cache.append(mask)
        bbox_list.append((float(x), float(y), float(w), float(h)))
        ann_ids.append(str(ann_id))
        pred_list.append(inst.predicted_iou)
        stab_list.append(inst.stability)

    if not instances:
        return []

    # Build neighbor union for visibility
    mask_arr = masks_cache
    n = len(instances)
    for i in range(n):
        box_i = bbox_list[i]
        short_side_i = min(box_i[2], box_i[3])
        cx_i = box_i[0] + 0.5 * box_i[2]
        cy_i = box_i[1] + 0.5 * box_i[3]
        neighbor_union = np.zeros_like(mask_arr[i], dtype=bool)
        for j in range(n):
            if i == j:
                continue
            box_j = bbox_list[j]
            # neighbor condition: bbox IoU>0 OR center distance < 0.5 * min(short sides)
            iou = bbox_iou_xywh(box_i, box_j)
            center_dist = math.hypot(cx_i - (box_j[0] + 0.5 * box_j[2]), cy_i - (box_j[1] + 0.5 * box_j[3]))
            short_side_j = min(box_j[2], box_j[3])
            dist_thresh = 0.5 * min(short_side_i, short_side_j)
            if iou > 0.0 or center_dist < dist_thresh:
                neighbor_union |= mask_arr[j]
        visible = mask_arr[i] & (~neighbor_union)
        visible_count = visible.sum()
        total_count = mask_arr[i].sum()
        instances[i].v_visible = 0.0 if total_count == 0 else float(visible_count / total_count)

    return instances


def passes_hard_gate(inst: InstanceMetrics, allow_fragment: bool) -> bool:
    a = inst.area_ratio
    if a < 0.002 or a > 0.20:
        return False
    if inst.predicted_iou < 0.90 or inst.stability < 0.90:
        return False
    if inst.solidity < 0.85 or inst.hole_frac > 0.10:
        return False
    if inst.aspect < 0.25 or inst.aspect > 4.0:
        return False
    if inst.cy < 0.25 or inst.cy > 0.85:
        return False
    if max(inst.touch_edges.get("top", 0.0), inst.touch_edges.get("bottom", 0.0)) > 0.60:
        return False
    v_threshold = 0.75 if allow_fragment else 0.85
    if inst.v_visible < v_threshold:
        return False
    if inst.delta_e < 0.0 or inst.er < 0.20:
        return False
    return True


def qualifies_fragment(inst: InstanceMetrics) -> bool:
    a = inst.area_ratio
    if a < 5e-4 or a > 0.02:
        return False
    if min(inst.predicted_iou, inst.stability) < 0.85:
        return False
    if inst.solidity < 0.75 or inst.hole_frac > 0.20:
        return False
    if inst.aspect < 0.2 or inst.aspect > 5.0:
        return False
    if inst.cy < 0.15 or inst.cy > 0.90:
        return False
    return True


def build_candidate(
    inst: InstanceMetrics,
    score_stats: Dict[str, float],
    cand_type: str,
    member_ids: Sequence[str],
    cand_id: Optional[str] = None,
) -> Candidate:
    cand_id = cand_id or inst.ann_id
    return Candidate(
        image_id=inst.image_id,
        cand_id=str(cand_id),
        type=cand_type,
        member_ids=[str(mid) for mid in member_ids],
        mask=inst.mask,
        area_px=inst.area_px,
        area_ratio=inst.area_ratio,
        bbox_xywh=inst.bbox_xywh,
        cx=inst.cx,
        cy=inst.cy,
        aspect=inst.aspect,
        solidity=inst.solidity,
        hole_frac=inst.hole_frac,
        touch_edges=inst.touch_edges,
        v_visible=inst.v_visible,
        er=inst.er,
        delta_e=inst.delta_e,
        d_in=inst.d_in,
        d_near=inst.d_near,
        r_tex=inst.r_tex,
        q_json=score_stats["q_json"],
        s_size=score_stats["s_size"],
        s_ctr=score_stats["s_ctr"],
        s_vis=score_stats["s_vis"],
        s_edge=score_stats["s_edge"],
        s_tex=score_stats["s_tex"],
        s_shape=score_stats["s_shape"],
        p_stuff=score_stats["p_stuff"],
        s_base=score_stats["s_base"],
        s_total=score_stats["s_total"],
        predicted_iou=inst.predicted_iou,
        stability=inst.stability,
    )


def process_image(
    json_path: Path,
    npz_dir: Path,
) -> Tuple[List[Candidate], List[Candidate], List[PairStats], List[ClusterGain], List[Candidate]]:
    data = load_json(json_path)
    stem = json_path.stem
    height = int(data["image"]["height"])
    width = int(data["image"]["width"])
    edge_map = load_edge_for_image(npz_dir, stem, (height, width))
    te = quantile(edge_map, 0.85)
    if edge_map.std() < EDGE_CONST_FALLBACK_STD:
        edge_map = np.full_like(edge_map, 0.5, dtype=np.float32)
    instances = compute_instances(data, edge_map, stem)
    if not instances:
        return [], [], [], [], []

    fragment_flags = [qualifies_fragment(inst) for inst in instances]
    hard_gate_flags = [passes_hard_gate(inst, allow_fragment=fragment_flags[idx]) for idx, inst in enumerate(instances)]

    filtered_instances: List[InstanceMetrics] = [inst for inst, ok in zip(instances, hard_gate_flags) if ok]
    filtered_fragment_flags = [frag for frag, ok in zip(fragment_flags, hard_gate_flags) if ok]

    if not filtered_instances:
        return [], [], [], [], []

    delta_e_array = np.array([inst.delta_e for inst in filtered_instances], dtype=np.float32)
    d_in_array = np.array([inst.d_in for inst in filtered_instances], dtype=np.float32)
    coverage_count = np.zeros_like(filtered_instances[0].mask, dtype=np.int32)
    for inst in filtered_instances:
        coverage_count += inst.mask.astype(np.int32)

    single_candidates: List[Candidate] = []
    for inst in filtered_instances:
        stats = compute_candidate_scores_single(inst, delta_e_array, d_in_array)
        cand = build_candidate(inst, stats, cand_type="single", member_ids=[inst.ann_id])
        single_candidates.append(cand)

    fragment_indices = [idx for idx, flag in enumerate(filtered_fragment_flags) if flag]
    fragment_candidates = [single_candidates[idx] for idx in fragment_indices]

    # 片对枚举
    pair_candidates_per_i: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    pair_distance_cache: Dict[Tuple[int, int], float] = {}
    selected_pairs: set[Tuple[int, int]] = set()

    for idx_a_pos, i in enumerate(fragment_indices):
        inst_i = filtered_instances[i]
        for j in fragment_indices[idx_a_pos + 1 :]:
            inst_j = filtered_instances[j]
            short_side_min = min(inst_i.short_side(), inst_j.short_side())
            d_max = min(64.0, 0.2 * short_side_min)
            _, _, d_ij = compute_nearest_boundary_points(inst_i.mask, inst_j.mask)
            if math.isinf(d_ij) or d_ij > d_max:
                continue
            pair_candidates_per_i[i].append((d_ij, j))
            pair_candidates_per_i[j].append((d_ij, i))
            pair_distance_cache[(i, j)] = d_ij

    for i in fragment_indices:
        if i not in pair_candidates_per_i:
            continue
        neighbors = sorted(pair_candidates_per_i[i], key=lambda x: x[0])
        for d, j in neighbors[:MAX_NEIGHBORS]:
            pair = (min(i, j), max(i, j))
            selected_pairs.add(pair)

    pair_stats_list: List[PairStats] = []
    pair_attempts: List[Tuple[float, int, int]] = []

    for i, j in sorted(selected_pairs):
        inst_i = filtered_instances[i]
        inst_j = filtered_instances[j]
        try:
            union_inst = build_union_instance([inst_i, inst_j], coverage_count, edge_map, te, stem)
        except ValueError:
            continue
        affinity, comps = compute_pair_affinity(inst_i, inst_j, union_inst, edge_map)
        if math.isinf(comps["d_ij"]):
            continue
        min_area = min(inst_i.area_px, inst_j.area_px)
        d_ij_norm = comps["d_ij"] / max(math.sqrt(min_area), 1.0)
        pair_stats_list.append(
            PairStats(
                image_id=stem,
                id_i=str(inst_i.ann_id),
                id_j=str(inst_j.ann_id),
                area_i=inst_i.area_px,
                area_j=inst_j.area_px,
                a_i=inst_i.area_ratio,
                a_j=inst_j.area_ratio,
                d_ij_px=comps["d_ij"],
                d_ij_norm=d_ij_norm,
                b_ij=comps["B_ij"],
                er_i=inst_i.er,
                er_j=inst_j.er,
                er_union=union_inst.er,
                d_er=comps["delta_er"],
                sol_i=inst_i.solidity,
                sol_j=inst_j.solidity,
                sol_union=union_inst.solidity,
                d_sol=comps["delta_sol"],
                hole_i=inst_i.hole_frac,
                hole_j=inst_j.hole_frac,
                hole_union=union_inst.hole_frac,
                d_hole=comps["delta_hole"],
                egap=comps["egap"],
                v_i=inst_i.v_visible,
                v_j=inst_j.v_visible,
                v_union=union_inst.v_visible,
                aspect_union=union_inst.aspect,
                a_ij=affinity,
            )
        )
        pair_attempts.append((affinity, i, j))

    # 受约束凝聚
    n = len(filtered_instances)
    parent = list(range(n))
    cluster_members: Dict[int, List[int]] = {idx: [idx] for idx in range(n)}
    cluster_instances: Dict[int, InstanceMetrics] = {idx: inst for idx, inst in enumerate(filtered_instances)}
    cluster_candidates: Dict[int, Candidate] = {idx: single_candidates[idx] for idx in range(n)}
    cluster_counter = 0
    cluster_gain_list: List[ClusterGain] = []

    def find_root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for affinity_init, i, j in sorted(pair_attempts, key=lambda x: x[0], reverse=True):
        ri = find_root(i)
        rj = find_root(j)
        if ri == rj:
            continue
        inst_i = cluster_instances[ri]
        inst_j = cluster_instances[rj]
        member_indices = cluster_members[ri] + cluster_members[rj]
        try:
            union_inst = build_union_instance(
                [filtered_instances[idx] for idx in member_indices],
                coverage_count,
                edge_map,
                te,
                stem,
            )
        except ValueError:
            continue
        affinity_current, comps_current = compute_pair_affinity(inst_i, inst_j, union_inst, edge_map)
        s_union_scores = compute_candidate_scores_single(union_inst, delta_e_array, d_in_array)
        member_ids = [filtered_instances[idx].ann_id for idx in cluster_members[ri] + cluster_members[rj]]
        union_candidate = build_candidate(
            union_inst,
            s_union_scores,
            cand_type="cluster",
            member_ids=member_ids,
            cand_id=f"{stem}_cluster_{cluster_counter}",
        )
        cluster_counter += 1

        s_i_base = cluster_candidates[ri].s_base
        s_j_base = cluster_candidates[rj].s_base
        d_s = union_candidate.s_base - max(s_i_base, s_j_base)
        cluster_gain_list.append(
            ClusterGain(
                image_id=stem,
                id_i="|".join(str(filtered_instances[idx].ann_id) for idx in cluster_members[ri]),
                id_j="|".join(str(filtered_instances[idx].ann_id) for idx in cluster_members[rj]),
                s_i_base=s_i_base,
                s_j_base=s_j_base,
                s_union_base=union_candidate.s_base,
                d_s=d_s,
                a_ij=affinity_current,
            )
        )

        # 合并条件
        short_side_min = min(inst_i.short_side(), inst_j.short_side())
        d_max = min(64.0, 0.2 * short_side_min)
        if comps_current["d_ij"] > d_max:
            continue
        if not (0.002 <= union_inst.area_ratio <= 0.20):
            continue
        if union_inst.solidity < max(inst_i.solidity, inst_j.solidity) - 0.05:
            continue
        if union_inst.hole_frac > max(inst_i.hole_frac, inst_j.hole_frac) + 0.05:
            continue
        if union_inst.er < max(inst_i.er, inst_j.er) - 0.05:
            continue
        if union_inst.v_visible < min(inst_i.v_visible, inst_j.v_visible) - 0.05:
            continue
        if max(union_inst.touch_edges.get("top", 0.0), union_inst.touch_edges.get("bottom", 0.0)) > 0.60:
            continue
        if d_s < TAU_DEFAULT:
            continue
        if affinity_current < THETA_DEFAULT:
            continue

        # 执行合并
        parent[rj] = ri
        cluster_members[ri] = cluster_members[ri] + cluster_members[rj]
        cluster_members.pop(rj, None)
        cluster_instances[ri] = union_inst
        cluster_candidates[ri] = union_candidate
        cluster_instances.pop(rj, None)
        cluster_candidates.pop(rj, None)
        for m in cluster_members[ri]:
            parent[m] = ri
        parent[ri] = ri

    seen_roots: set[int] = set()
    final_candidates: List[Candidate] = []
    for idx in range(n):
        root = find_root(idx)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        final_candidates.append(cluster_candidates[root])

    return single_candidates, fragment_candidates, pair_stats_list, cluster_gain_list, final_candidates


def write_pairs_csv(path: Path, pairs: Sequence[PairStats]) -> None:
    header = [
        "image_id",
        "id_i",
        "id_j",
        "area_i",
        "area_j",
        "a_i",
        "a_j",
        "d_ij_px",
        "d_ij_norm",
        "B_ij",
        "ER_i",
        "ER_j",
        "ER_union",
        "dER",
        "sol_i",
        "sol_j",
        "sol_union",
        "dsol",
        "hole_i",
        "hole_j",
        "hole_union",
        "dhole",
        "Egap",
        "v_i",
        "v_j",
        "v_union",
        "aspect_union",
        "A_ij",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for p in pairs:
            writer.writerow([
                p.image_id,
                p.id_i,
                p.id_j,
                p.area_i,
                p.area_j,
                f"{p.a_i:.6f}",
                f"{p.a_j:.6f}",
                f"{p.d_ij_px:.4f}",
                f"{p.d_ij_norm:.4f}",
                f"{p.b_ij:.6f}",
                f"{p.er_i:.6f}",
                f"{p.er_j:.6f}",
                f"{p.er_union:.6f}",
                f"{p.d_er:.6f}",
                f"{p.sol_i:.6f}",
                f"{p.sol_j:.6f}",
                f"{p.sol_union:.6f}",
                f"{p.d_sol:.6f}",
                f"{p.hole_i:.6f}",
                f"{p.hole_j:.6f}",
                f"{p.hole_union:.6f}",
                f"{p.d_hole:.6f}",
                f"{p.egap:.6f}",
                f"{p.v_i:.6f}",
                f"{p.v_j:.6f}",
                f"{p.v_union:.6f}",
                f"{p.aspect_union:.6f}",
                f"{p.a_ij:.6f}",
                p.label,
            ])


def write_cluster_gain_csv(path: Path, gains: Sequence[ClusterGain]) -> None:
    header = [
        "image_id",
        "id_i",
        "id_j",
        "S_i_base",
        "S_j_base",
        "S_union_base",
        "dS",
        "A_ij",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for g in gains:
            writer.writerow([
                g.image_id,
                g.id_i,
                g.id_j,
                f"{g.s_i_base:.6f}",
                f"{g.s_j_base:.6f}",
                f"{g.s_union_base:.6f}",
                f"{g.d_s:.6f}",
                f"{g.a_ij:.6f}",
                g.label,
            ])


def write_candidate_stats_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    header = [
        "image_id",
        "cand_id",
        "type",
        "member_ids",
        "area_px",
        "a_ratio",
        "cx",
        "cy",
        "r_aspect",
        "sol",
        "hole_frac",
        "v",
        "ER",
        "DeltaE",
        "d_in",
        "d_near",
        "R_tex",
        "Q_json",
        "S_size",
        "S_ctr",
        "S_vis",
        "S_edge",
        "S_tex",
        "S_shape",
        "P_stuff",
        "S_base",
        "S_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            writer.writerow([
                c.image_id,
                c.cand_id,
                c.type,
                "|".join(map(str, c.member_ids)),
                c.area_px,
                f"{c.area_ratio:.6f}",
                f"{c.cx:.6f}",
                f"{c.cy:.6f}",
                f"{c.aspect:.6f}",
                f"{c.solidity:.6f}",
                f"{c.hole_frac:.6f}",
                f"{c.v_visible:.6f}",
                f"{c.er:.6f}",
                f"{c.delta_e:.6f}",
                f"{c.d_in:.6f}",
                f"{c.d_near:.6f}",
                f"{c.r_tex:.6f}",
                f"{c.q_json:.6f}",
                f"{c.s_size:.6f}",
                f"{c.s_ctr:.6f}",
                f"{c.s_vis:.6f}",
                f"{c.s_edge:.6f}",
                f"{c.s_tex:.6f}",
                f"{c.s_shape:.6f}",
                f"{c.p_stuff:.6f}",
                f"{c.s_base:.6f}",
                f"{c.s_total:.6f}",
            ])

def compute_nearest_boundary_points(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    """返回两个掩码的最近边界像素坐标以及像素距离"""
    contours_a, _ = cv2.findContours(mask_a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_b, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_a or not contours_b:
        return (0, 0), (0, 0), float("inf")
    pts_a = np.vstack(contours_a).squeeze(1)
    pts_b = np.vstack(contours_b).squeeze(1)
    # KD-tree 加速可能更优，这里直接暴力（点数通常较少）
    min_dist = float("inf")
    min_pt_a = (0, 0)
    min_pt_b = (0, 0)
    for ax, ay in pts_a:
        diff = pts_b - np.array([ax, ay])
        dists = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dists))
        dist_val = dists[idx]
        if dist_val < min_dist:
            min_dist = dist_val
            min_pt_a = (int(ax), int(ay))
            bx, by = pts_b[idx]
            min_pt_b = (int(bx), int(by))
    return min_pt_a, min_pt_b, float(math.sqrt(min_dist))


def draw_line_band(shape: Tuple[int, int], pt_a: Tuple[int, int], pt_b: Tuple[int, int], thickness: int = 3) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.line(canvas, pt_a, pt_b, color=1, thickness=max(1, thickness))
    return canvas.astype(bool)


def compute_union_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= ensure_bool(m)
    return union


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------

@dataclass
class InstanceMetrics:
    image_id: str
    ann_id: int | str
    mask: np.ndarray  # bool
    area_px: int
    bbox_xywh: Tuple[float, float, float, float]
    cx: float
    cy: float
    aspect: float
    solidity: float
    hole_frac: float
    touch_edges: Dict[str, float]
    predicted_iou: float
    stability: float
    v_visible: float
    delta_e: float
    er: float
    er_in: float
    er_out: float
    d_in: float
    d_near: float
    r_tex: float
    area_ratio: float
    edge_quantile: float

    def short_side(self) -> float:
        _, _, w, h = self.bbox_xywh
        return min(w, h)

    def long_side(self) -> float:
        _, _, w, h = self.bbox_xywh
        return max(w, h)

    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return bbox_to_xyxy(self.bbox_xywh)


@dataclass
class Candidate:
    image_id: str
    cand_id: str
    type: str  # "single" or "cluster"
    member_ids: List[str]
    mask: np.ndarray  # bool
    area_px: int
    area_ratio: float
    bbox_xywh: Tuple[float, float, float, float]
    cx: float
    cy: float
    aspect: float
    solidity: float
    hole_frac: float
    touch_edges: Dict[str, float]
    v_visible: float
    er: float
    delta_e: float
    d_in: float
    d_near: float
    r_tex: float
    q_json: float
    s_size: float
    s_ctr: float
    s_vis: float
    s_edge: float
    s_tex: float
    s_shape: float
    p_stuff: float
    s_base: float
    s_total: float
    predicted_iou: float
    stability: float

    def short_side(self) -> float:
        _, _, w, h = self.bbox_xywh
        return min(w, h)

    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return bbox_to_xyxy(self.bbox_xywh)


@dataclass
class PairStats:
    image_id: str
    id_i: str
    id_j: str
    area_i: int
    area_j: int
    a_i: float
    a_j: float
    d_ij_px: float
    d_ij_norm: float
    b_ij: float
    er_i: float
    er_j: float
    er_union: float
    d_er: float
    sol_i: float
    sol_j: float
    sol_union: float
    d_sol: float
    hole_i: float
    hole_j: float
    hole_union: float
    d_hole: float
    egap: float
    v_i: float
    v_j: float
    v_union: float
    aspect_union: float
    a_ij: float
    label: str = ""


@dataclass
class ClusterGain:
    image_id: str
    id_i: str
    id_j: str
    s_i_base: float
    s_j_base: float
    s_union_base: float
    d_s: float
    a_ij: float
    label: str = ""


# -----------------------------------------------------------------------------
# 合体 & 片对辅助
# -----------------------------------------------------------------------------


def aggregate_quality(members: Sequence[InstanceMetrics]) -> Tuple[float, float]:
    total_area = sum(inst.area_px for inst in members)
    if total_area <= 0:
        return 0.0, 0.0
    pred = sum(inst.predicted_iou * inst.area_px for inst in members) / total_area
    stab = sum(inst.stability * inst.area_px for inst in members) / total_area
    return float(pred), float(stab)


def build_union_instance(
    members: Sequence[InstanceMetrics],
    coverage_count: np.ndarray,
    edge_map: np.ndarray,
    te: float,
    image_id: str,
) -> InstanceMetrics:
    if not members:
        raise ValueError("union members empty")
    mask_union = compute_union_mask([inst.mask for inst in members])
    area_px = int(mask_union.sum())
    if area_px == 0:
        raise ValueError("union mask empty")
    h, w = mask_union.shape
    total_pixels = h * w
    ys, xs = np.where(mask_union)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_xywh = (float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1))
    cx = (bbox_xywh[0] + 0.5 * bbox_xywh[2]) / w
    cy = (bbox_xywh[1] + 0.5 * bbox_xywh[3]) / h
    aspect = max(bbox_xywh[2], bbox_xywh[3]) / max(min(bbox_xywh[2], bbox_xywh[3]), EPS)

    solidity = 0.0
    hull_area = compute_convex_hull_area(mask_union)
    if hull_area > 0:
        solidity = float(area_px / hull_area)

    hole_frac = compute_hole_fraction(mask_union)
    touch_edges = touch_edge_ratios(mask_union)
    edge_stats = compute_edge_regions(mask_union, compute_signed_distance(mask_union), edge_map, te)

    member_cover = np.zeros_like(coverage_count, dtype=np.int32)
    for inst in members:
        member_cover += inst.mask.astype(np.int32)
    other_cover = coverage_count - member_cover
    other_cover = np.maximum(other_cover, 0)
    visible = mask_union & (other_cover == 0)
    v_visible = 0.0 if area_px == 0 else float(visible.sum() / max(area_px, 1))

    pred, stab = aggregate_quality(members)

    return InstanceMetrics(
        image_id=image_id,
        ann_id="+".join(str(inst.ann_id) for inst in members),
        mask=mask_union,
        area_px=area_px,
        bbox_xywh=bbox_xywh,
        cx=float(cx),
        cy=float(cy),
        aspect=float(aspect),
        solidity=float(solidity),
        hole_frac=float(hole_frac),
        touch_edges=touch_edges,
        predicted_iou=pred,
        stability=stab,
        v_visible=v_visible,
        delta_e=edge_stats["delta_e"],
        er=edge_stats["ER"],
        er_in=edge_stats["E_in"],
        er_out=edge_stats["E_out"],
        d_in=edge_stats["d_in"],
        d_near=edge_stats["d_near"],
        r_tex=edge_stats["r_tex"],
        area_ratio=float(area_px / total_pixels),
        edge_quantile=te,
    )


def compute_bridge_ratio(mask_union: np.ndarray) -> float:
    union_uint8 = mask_union.astype(np.uint8)
    closed = cv2.morphologyEx(union_uint8, cv2.MORPH_CLOSE, make_disk(R_CLOSE))
    closed_bool = closed.astype(bool)
    denom = closed_bool.sum()
    if denom == 0:
        return 0.0
    bridge = closed_bool & (~mask_union)
    return float(bridge.sum() / denom)


def compute_egap(mask_i: np.ndarray, mask_j: np.ndarray, edge_map: np.ndarray) -> float:
    pt_a, pt_b, dist = compute_nearest_boundary_points(mask_i, mask_j)
    if math.isinf(dist):
        return 0.0
    band = draw_line_band(edge_map.shape, pt_a, pt_b, thickness=LINE_THICKNESS)
    if dist < 3.0:
        band = cv2.dilate(band.astype(np.uint8), make_disk(3)).astype(bool)
    if band.sum() == 0:
        return 0.0
    return float(edge_map[band].mean())


def compute_pair_affinity(
    inst_i: InstanceMetrics,
    inst_j: InstanceMetrics,
    union_inst: InstanceMetrics,
    edge_map: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    mask_i = inst_i.mask
    mask_j = inst_j.mask
    d_pt_i, d_pt_j, d_ij = compute_nearest_boundary_points(mask_i, mask_j)
    if math.isinf(d_ij):
        d_ij = 1e6
    lambda_val = 1.5 * math.sqrt(min(inst_i.area_px, inst_j.area_px))
    lambda_val = max(lambda_val, 1.0)
    D = math.exp(-d_ij / lambda_val)

    B_ij = compute_bridge_ratio(mask_i | mask_j)
    C = math.exp(-B_ij / 0.10)

    delta_er = union_inst.er - max(inst_i.er, inst_j.er)
    egap = compute_egap(mask_i, mask_j, edge_map)
    K = sigmoid(2.0 * delta_er + 1.5 * (egap - 0.5))

    delta_sol = union_inst.solidity - max(inst_i.solidity, inst_j.solidity)
    delta_hole = min(inst_i.hole_frac, inst_j.hole_frac) - union_inst.hole_frac
    aspect_penalty_union = max(0.0, abs(math.log(union_inst.aspect + EPS)) - math.log(2.5)) / math.log(2.5)
    G = sigmoid(3.0 * delta_sol + 2.0 * delta_hole - aspect_penalty_union)

    V = 1.0 if union_inst.v_visible >= min(inst_i.v_visible, inst_j.v_visible) - 0.05 else 0.0

    A = (D ** 0.25) * (C ** 0.20) * (K ** 0.35) * (G ** 0.40) * V

    return float(A), {
        "D": D,
        "C": C,
        "K": K,
        "G": G,
        "V": V,
        "B_ij": B_ij,
        "delta_er": delta_er,
        "delta_sol": delta_sol,
        "delta_hole": delta_hole,
        "egap": egap,
        "d_ij": d_ij,
    }


# -----------------------------------------------------------------------------
# 主体逻辑（占位实现）
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SA-1B 候选体统计脚本")
    parser.add_argument("--json-dir", type=Path, required=True, help="SA-1B JSON 目录（包含 sa_*.json）")
    parser.add_argument("--npz-dir", type=Path, required=True, help="NPZ 目录（包含 sa_*_features.npz）")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--num-samples", type=int, default=10000, help="采样图像数量")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--sigma_edge_min_iqr", type=float, default=1e-3, help="S_edge IQR 下限")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    json_files = sorted([p for p in args.json_dir.glob("sa_*.json") if p.is_file()])
    if not json_files:
        raise FileNotFoundError(f"未在 {args.json_dir} 找到任何 sa_*.json")

    if args.num_samples > len(json_files):
        selected_files = json_files
    else:
        selected_files = random.sample(json_files, args.num_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_single: List[Candidate] = []
    all_fragment: List[Candidate] = []
    all_pairs: List[PairStats] = []
    all_cluster_gain: List[ClusterGain] = []
    all_final_candidates: List[Candidate] = []

    for idx, json_path in enumerate(selected_files, start=1):
        try:
            single, fragment, pairs, gains, final_cands = process_image(json_path, args.npz_dir)
        except Exception as exc:
            print(f"[{idx}/{len(selected_files)}] 处理 {json_path.name} 失败: {exc}")
            continue
        all_single.extend(single)
        all_fragment.extend(fragment)
        all_pairs.extend(pairs)
        all_cluster_gain.extend(gains)
        all_final_candidates.extend(final_cands)
        if idx % 100 == 0 or idx == len(selected_files):
            print(f"[{idx}/{len(selected_files)}] 已处理")

    print(f"单片候选: {len(all_single)}，碎片候选: {len(all_fragment)}，片对: {len(all_pairs)}，合体增益: {len(all_cluster_gain)}，候选体统计: {len(all_final_candidates)}")

    pairs_path = args.output_dir / "pairs_stats.csv"
    cluster_gain_path = args.output_dir / "cluster_gain.csv"
    candidate_stats_path = args.output_dir / "candidate_stats.csv"

    write_pairs_csv(pairs_path, all_pairs)
    write_cluster_gain_csv(cluster_gain_path, all_cluster_gain)
    write_candidate_stats_csv(candidate_stats_path, all_final_candidates)

    print(f"已写出 CSV：\n - {pairs_path}\n - {cluster_gain_path}\n - {candidate_stats_path}")


if __name__ == "__main__":
    main()


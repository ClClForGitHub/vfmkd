import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

try:
    # 可选依赖，用于RLE分割掩码解码
    from pycocotools import mask as maskUtils  # type: ignore
except Exception:  # pragma: no cover
    maskUtils = None

try:
    import cv2
except ImportError:
    cv2 = None


def load_sa_json(json_path: str | Path) -> Dict:
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_bboxes_xywh(sa_obj: Dict) -> Tuple[int, int, List[List[float]]]:
    """
    从SA风格json抽取图像尺寸与bbox(x,y,w,h)列表。
    期望结构: {
      "image": {"width": W, "height": H, ...},
      "annotations": [{"bbox": [x,y,w,h], ...}, ...]
    }
    """
    img = sa_obj.get("image", {})
    width = int(img.get("width", 0))
    height = int(img.get("height", 0))
    ann = sa_obj.get("annotations", [])
    boxes = []
    for a in ann:
        b = a.get("bbox", None)
        if isinstance(b, list) and len(b) == 4:
            boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return width, height, boxes


def xywh_to_xyxy(boxes_xywh: List[List[float]]) -> torch.Tensor:
    if len(boxes_xywh) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    b = torch.tensor(boxes_xywh, dtype=torch.float32)
    x1y1 = b[:, :2]
    x2y2 = b[:, :2] + b[:, 2:4]
    return torch.cat([x1y1, x2y2], dim=1)


def build_batch_bboxes_from_ids(
    image_ids: List[str], json_dir: str | Path
) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """
    根据 image_id 列表在 json_dir 查找 `sa_{image_id}.json`，返回：
      - bboxes_xyxy: 每张图的N×4张量（原图坐标系）
      - 统一 img_size (H,W): 若各图尺寸不同，返回第一张的尺寸（调用方可覆盖）
    若找不到json或无标注，返回空框张量。
    """
    bboxes_list: List[torch.Tensor] = []
    img_size: Tuple[int, int] | None = None
    json_dir = Path(json_dir)
    for img_id in image_ids:
        candidate = json_dir / f"sa_{img_id}.json"
        if not candidate.exists():
            bboxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
            continue
        obj = load_sa_json(candidate)
        w, h, boxes_xywh = parse_bboxes_xywh(obj)
        if img_size is None:
            img_size = (h, w)
        bboxes_xyxy = xywh_to_xyxy(boxes_xywh)
        bboxes_list.append(bboxes_xyxy)
    if img_size is None:
        img_size = (1024, 1024)
    return bboxes_list, img_size


def _decode_union_mask_from_json(sa_obj: Dict) -> Optional[torch.Tensor]:
    """
    将单张JSON的所有实例分割合并为一个二值mask (H,W)。
    优先使用 segmentation RLE，若不可用则回退为空。
    """
    if maskUtils is None:
        return None
    ann = sa_obj.get("annotations", [])
    if not ann:
        return torch.zeros(0)
    H, W = sa_obj.get("image", {}).get("height", None), sa_obj.get("image", {}).get("width", None)
    # 一些JSON里segmentation.size即为(H,W)，更稳妥：优先取segmentation.size
    masks = []
    for a in ann:
        seg = a.get("segmentation", None)
        if not seg:
            continue
        size = seg.get("size", None)
        counts = seg.get("counts", None)
        if not size or counts is None:
            continue
        rle = {"size": size, "counts": counts}
        m = maskUtils.decode(rle)  # (H,W,1)或(H,W)
        if m.ndim == 3:
            m = m[:, :, 0]
        masks.append(torch.from_numpy(m.astype("uint8")))
    if not masks:
        # 无segmentation时回退
        if H is None or W is None:
            return torch.zeros(0)
        return torch.zeros((H, W), dtype=torch.uint8)
    union = masks[0].clone()
    for m in masks[1:]:
        union = torch.maximum(union, m)
    return union  # (H,W) uint8 {0,1}


def build_fg_bg_from_ids(
    image_ids: List[str], json_dir: str | Path, feat_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 sa_{image_id}.json 读取合并分割掩码，生成前景/背景权重图，
    下采样到特征图分辨率 feat_size=(Hf,Wf)。

    前景权重：二值前景经自适应平均池化至(Hf,Wf)，得到[0,1]概率，再归一化常数area权重: 1/NumFgPixels。
    背景权重：1-FgBin 后归一化为概率（sum=1）。
    """
    Hf, Wf = feat_size
    json_dir = Path(json_dir)
    fg_list: List[torch.Tensor] = []
    bg_list: List[torch.Tensor] = []
    for img_id in image_ids:
        p = json_dir / f"sa_{img_id}.json"
        if not p.exists():
            # 无标注：前景全0，背景全1(归一后)
            fg_bin = torch.zeros((Hf, Wf), dtype=torch.float32)
            bg = torch.ones((Hf, Wf), dtype=torch.float32)
            bg /= bg.sum()
            fg_list.append(fg_bin)
            bg_list.append(bg)
            continue
        obj = load_sa_json(p)
        union = _decode_union_mask_from_json(obj)
        if union is None or union.numel() == 0:
            # 解码失败或无掩码
            fg_bin = torch.zeros((Hf, Wf), dtype=torch.float32)
            bg = torch.ones((Hf, Wf), dtype=torch.float32)
            bg /= bg.sum()
            fg_list.append(fg_bin)
            bg_list.append(bg)
            continue
        # 下采样为(Hf,Wf)的概率图
        m = union.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        fg_prob = torch.nn.functional.adaptive_avg_pool2d(m, (Hf, Wf)).squeeze(0).squeeze(0)  # (Hf,Wf)
        # area常数前景（FGD风格）
        num_fg = torch.clamp((fg_prob > 0.5).float().sum(), min=1)
        fg_area = (fg_prob > 0.5).float() / num_fg
        fg_list.append(fg_area)
        # 背景概率并归一
        bg = 1.0 - (fg_prob > 0.5).float()
        s = bg.sum()
        if s > 0:
            bg = bg / s
        bg_list.append(bg)

    fg_stack = torch.stack(fg_list, dim=0).unsqueeze(1)  # (B,1,Hf,Wf)
    bg_stack = torch.stack(bg_list, dim=0).unsqueeze(1)
    return fg_stack, bg_stack


def load_edge_maps_from_npz(
    image_ids: List[str], 
    npz_dir: str | Path, 
    feat_size: Tuple[int, int],
    use_real_edge: bool = False,
    real_edge_maps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    加载边缘图并下采样到特征图分辨率
    
    Args:
        image_ids: 图像ID列表
        npz_dir: NPZ文件目录（用于模拟边缘图）
        feat_size: 目标特征图尺寸 (Hf, Wf)
        use_real_edge: 是否使用真实边缘图（未来接口）
        real_edge_maps: 真实边缘图 (B, 1, Hf, Wf)，如果提供则直接返回
    
    Returns:
        torch.Tensor: (B, 1, Hf, Wf) 边缘图，边缘=1, 非边缘=0
    """
    # 【真实边缘图接口】如果提供了真实边缘图，直接返回
    if use_real_edge and real_edge_maps is not None:
        return real_edge_maps
    
    # 【模拟边缘图】从NPZ的edge_256x256下采样
    if cv2 is None:
        raise ImportError("cv2 is required for edge map downsampling")
    
    Hf, Wf = feat_size
    npz_dir = Path(npz_dir)
    edge_list: List[torch.Tensor] = []
    
    for img_id in image_ids:
        npz_file = npz_dir / f"sa_{img_id}_features.npz"
        
        if not npz_file.exists():
            # 无NPZ文件：边缘全0
            edge_list.append(torch.zeros((Hf, Wf), dtype=torch.float32))
            continue
        
        try:
            data = np.load(npz_file)
            if 'edge_256x256' not in data:
                edge_list.append(torch.zeros((Hf, Wf), dtype=torch.float32))
                continue
            
            edge_256 = data['edge_256x256']  # (256, 256) uint8 {0, 1}
            
            # 【与extract_features_v1.py完全一致的下采样方法】
            # 1. 转为float32
            edge_float = edge_256.astype(np.float32)
            
            # 2. cv2.resize + INTER_AREA
            edge_resized_float = cv2.resize(
                edge_float,
                (Wf, Hf),  # 注意：cv2.resize是(width, height)
                interpolation=cv2.INTER_AREA
            )
            
            # 3. 阈值化：> 0 则为 1
            edge_binary = (edge_resized_float > 0).astype(np.float32)
            
            edge_list.append(torch.from_numpy(edge_binary))
            
        except Exception as e:
            print(f"[WARN] 加载边缘图失败 {img_id}: {e}")
            edge_list.append(torch.zeros((Hf, Wf), dtype=torch.float32))
    
    # Stack并添加通道维度
    edge_stack = torch.stack(edge_list, dim=0).unsqueeze(1)  # (B, 1, Hf, Wf)
    return edge_stack


def apply_edge_boost(
    fg_map: torch.Tensor, 
    edge_map: torch.Tensor, 
    boost_factor: float = 3.0
) -> torch.Tensor:
    """
    在前景mask的边缘位置增强权重
    
    Args:
        fg_map: 原始前景权重 (B, 1, H, W)
        edge_map: 边缘图 (B, 1, H, W), 边缘=1, 非边缘=0
        boost_factor: 增强因子，3.0表示边缘梯度≈2倍前景（因为sqrt(1+3)=2）
    
    Returns:
        torch.Tensor: 增强后的前景mask (B, 1, H, W)
    """
    return fg_map + boost_factor * edge_map


def load_weights_from_npz(
    image_ids: List[str],
    npz_dir: str | Path,
    feat_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从NPZ文件加载预计算的前景/背景权重图
    
    Args:
        image_ids: 图像ID列表
        npz_dir: NPZ文件所在目录
        feat_size: 特征图尺寸 (H, W), 例如 (64, 64)
    
    Returns:
        fg_map: 前景权重 (B, 1, H, W)
        bg_map: 背景权重 (B, 1, H, W)
    """
    npz_path = Path(npz_dir)
    Hf, Wf = feat_size
    
    # 确定键名（根据特征尺寸）
    fg_key = f'fg_map_{Hf}x{Wf}'
    bg_key = f'bg_map_{Hf}x{Wf}'
    
    fg_list = []
    bg_list = []
    
    for img_id in image_ids:
        # 尝试多种可能的文件名格式
        possible_names = [
            f"sa_{img_id}_features.npz",
            f"{img_id}_features.npz"
        ]
        
        npz_file = None
        for name in possible_names:
            candidate = npz_path / name
            if candidate.exists():
                npz_file = candidate
                break
        
        if npz_file is None:
            raise FileNotFoundError(f"NPZ file not found for image_id: {img_id}")
        
        # 加载NPZ
        data = np.load(npz_file)
        
        # 检查是否有预计算的权重
        if fg_key not in data or bg_key not in data:
            raise KeyError(f"Precomputed weights '{fg_key}' or '{bg_key}' not found in {npz_file.name}")
        
        # 加载权重
        fg = torch.from_numpy(data[fg_key]).float()  # (H, W)
        bg = torch.from_numpy(data[bg_key]).float()  # (H, W)
        
        fg_list.append(fg)
        bg_list.append(bg)
    
    # Stack并添加通道维度
    fg_stack = torch.stack(fg_list, dim=0).unsqueeze(1)  # (B, 1, H, W)
    bg_stack = torch.stack(bg_list, dim=0).unsqueeze(1)  # (B, 1, H, W)
    
    return fg_stack, bg_stack



#!/usr/bin/env python3
"""
Panel-A 单图运行器：
- 加载SAM2头（Base+）与transforms
- 加载学生backbone+adapter权重，提取学生P4(1,256,64,64)
- 加载教师NPZ并对齐，提取实时教师P4
- 统一提示为整图框，使用原生prompt/decoder生成三张掩码
- 计算三对特征相似度与三对掩码IOU
- 绘制两行七图面板
"""

from pathlib import Path
from typing import Optional, Dict
import sys

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.core.prompt.heads import load_sam2_heads
from tools.core.prompt.student_loader import (
    load_student_backbone_and_adapter,
    extract_student_p4_from_image,
    align_student_features,
)
from tools.core.prompt.teacher_io import load_teacher_npz, extract_realtime_teacher_p4
from tools.core.prompt.metrics import compute_similarity, mask_iou
from tools.core.prompt.panel_viz import draw_panel


@torch.no_grad()
def run_panel(
    image_path: Path,
    student_weights: Path,
    teacher_npz_path: Path,
    sam2_cfg: str = 'sam2.1/sam2.1_hiera_b+.yaml',
    sam2_ckpt: Optional[Path] = None,
    device: str = 'cuda:6',
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 1) SAM2头与预处理
    pe, md, trans, sam2 = load_sam2_heads(
        device=device,
        config_file=sam2_cfg,
        ckpt_path=str(sam2_ckpt) if sam2_ckpt else None,
        return_model=True,
    )

    # 2) 读取图片 & 预处理
    image_pil = Image.open(str(image_path)).convert('RGB')
    image_np = np.array(image_pil)
    image_tensor_display = TF.to_tensor(image_pil.resize((1024, 1024)))  # (3,1024,1024)
    image_tensor_1024 = trans(image_np).unsqueeze(0).to(device_obj)

    # 学生backbone+adapter 与原始S16特征
    backbone, adapter = load_student_backbone_and_adapter(student_weights, device=device)
    student_s16, feats_for_adapter = extract_student_p4_from_image(image_path, backbone, device=device)

    # 3) 教师：NPZ 与 实时
    feat_npz = load_teacher_npz(teacher_npz_path, device=device)
    feat_teacher = extract_realtime_teacher_p4(sam2, image_tensor_1024)

    # 4) 使用适配器将学生特征对齐到教师空间（优先NPZ）
    alignment_target = feat_npz if feat_npz is not None else feat_teacher
    feat_student = align_student_features(student_s16, feats_for_adapter, alignment_target, adapter)

    # 4) 统一提示：整图框
    H, W = image_tensor_1024.shape[-2:]
    boxes = torch.tensor([[0.0, 0.0, float(W), float(H)]], device=device_obj)  # (1, 4) XYXY格式

    # prompt编码
    # SAM2 prompt encoder 返回元组 (sparse_embeddings, dense_embeddings)
    sparse_embeddings, dense_embeddings = pe(points=None, boxes=boxes, masks=None)

    # 掩码解码函数
    # image_pe 需要从 prompt_encoder 获取
    image_pe = pe.get_dense_pe()
    
    def decode_mask(feat_p4: torch.Tensor):
        return md(
            image_embeddings=feat_p4,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
        )

    out_s = decode_mask(feat_student)
    out_n = decode_mask(feat_npz)
    out_t = decode_mask(feat_teacher)

    def to_mask_dict(out):
        # mask_decoder 返回元组: (low_res_multimasks, ious, sam_output_tokens, object_score_logits)
        if isinstance(out, tuple):
            logits = out[0]  # low_res_multimasks: (B, 1, H, W) 或 (B, M, H, W)
            # 如果是 multimask，取第一个
            if logits.dim() == 4 and logits.shape[1] > 1:
                logits = logits[:, 0:1]  # 取第一个mask
        else:
            raise RuntimeError(f'mask decoder返回格式不支持: {type(out)}')
        prob = torch.sigmoid(logits)
        binary = (prob > 0.5).float()
        return {'logits': logits, 'prob': prob, 'binary': binary}

    m_s = to_mask_dict(out_s)
    m_n = to_mask_dict(out_n)
    m_t = to_mask_dict(out_t)

    # 5) 特征相似度 + 掩码IOU
    sim_sn = compute_similarity(feat_student, feat_npz)
    sim_st = compute_similarity(feat_student, feat_teacher)
    sim_nt = compute_similarity(feat_npz, feat_teacher)
    sims = {
        'cos_student_npz': sim_sn['cosine'],
        'mae_student_npz': sim_sn['mae'],
        'l2_student_npz': sim_sn['l2'],
        'cos_student_teacher': sim_st['cosine'],
        'mae_student_teacher': sim_st['mae'],
        'l2_student_teacher': sim_st['l2'],
        'cos_npz_teacher': sim_nt['cosine'],
        'mae_npz_teacher': sim_nt['mae'],
        'l2_npz_teacher': sim_nt['l2'],
    }
    ious = {
        'iou_student_npz': mask_iou(m_s['binary'], m_n['binary']),
        'iou_student_teacher': mask_iou(m_s['binary'], m_t['binary']),
        'iou_npz_teacher': mask_iou(m_n['binary'], m_t['binary']),
    }

    # 6) 绘制面板
    if output_path is None:
        output_path = _ROOT / 'outputs' / 'panel_a.png'
    output_path = Path(output_path)
    draw_panel(
        image_tensor_0_1=image_tensor_display,
        feat_student=feat_student,
        feat_npz=feat_npz,
        feat_teacher=feat_teacher,
        mask_student=m_s,
        mask_npz=m_n,
        mask_teacher=m_t,
        sim_metrics=sims,
        mask_ious=ious,
        output_path=output_path,
    )

    # 返回关键指标，便于外部记录
    return {**sims, **ious, 'output': str(output_path)}


__all__ = ['run_panel']



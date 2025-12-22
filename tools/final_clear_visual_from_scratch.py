#!/usr/bin/env python3
"""从头生成清晰的掩码可视化 - 直接使用原始数据"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear
        sys.modules[_mod] = stub_mod

from vfmkd.models.backbones.repvit_backbone import RepViTBackbone
from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
config = {'arch': 'm1', 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None}
backbone = RepViTBackbone(config).to(device).eval()
adapter = RepViTAlignAdapter(image_size=1024, hidden_dim=256).to(device).eval()

ckpt = torch.load('outputs/overfit_single_image_weights.pth', map_location='cpu', weights_only=False)
backbone.load_state_dict(ckpt['backbone'])
adapter.load_state_dict(ckpt['adapter'])

# 加载图片
img_pil = Image.open('datasets/coco128/images/train2017/000000000009.jpg').convert('RGB').resize((1024, 1024))
x = TF.to_tensor(img_pil).unsqueeze(0).to(device)

# 前向
with torch.no_grad():
    feat = backbone(x)
    adapted = adapter(feat)

# 加载SAM2
sam2_config_dir = _ROOT / "vfmkd" / "sam2" / "sam2" / "configs"
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
    sam2_model = build_sam2(
        config_file='sam2.1/sam2.1_hiera_b+.yaml',
        ckpt_path='weights/sam2.1_hiera_base_plus.pt',
        device=str(device)
    )

sam2_model.eval()
prompt_encoder = sam2_model.sam_prompt_encoder
mask_decoder = sam2_model.sam_mask_decoder
mask_decoder.use_high_res_features = False

# 准备prompt
point = [250, 400]
point_coords = torch.tensor([[point]], dtype=torch.float32, device=device)
point_labels = torch.tensor([[1]], dtype=torch.int32, device=device)

sparse_embeddings, dense_embeddings = prompt_encoder(
    points=(point_coords, point_labels),
    boxes=None,
    masks=None,
)

# RepViT+Adapter生成掩码
low_res_masks_repvit, iou_pred_repvit, _, _ = mask_decoder(
    image_embeddings=adapted,
    image_pe=prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,
    repeat_image=False,
    high_res_features=None,
)

# SAM2 Native生成掩码
# 加载teacher特征
npz = np.load('datasets/coco128/SAM_Cache/000000000009_sam2_features.npz')
if 'IMAGE_EMB_S16' in npz:
    teacher_feat = torch.from_numpy(npz['IMAGE_EMB_S16']).to(device)
elif 'image_embedding' in npz:
    teacher_feat = torch.from_numpy(npz['image_embedding']).to(device)

low_res_masks_sam2, iou_pred_sam2, _, _ = mask_decoder(
    image_embeddings=teacher_feat,
    image_pe=prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,
    repeat_image=False,
    high_res_features=None,
)

print(f"[INFO] RepViT IOU: {iou_pred_repvit[0].detach().cpu().numpy()}")
print(f"[INFO] SAM2 IOU: {iou_pred_sam2[0].detach().cpu().numpy()}")

# 选择最佳掩码
best_idx_repvit = iou_pred_repvit[0].argmax().item()
best_idx_sam2 = iou_pred_sam2[0].argmax().item()

# 上采样到1024x1024
mask_repvit_1024 = F.interpolate(
    low_res_masks_repvit[:, best_idx_repvit:best_idx_repvit+1],
    size=(1024, 1024),
    mode='bilinear',
    align_corners=False
)[0, 0].cpu().numpy()

mask_sam2_1024 = F.interpolate(
    low_res_masks_sam2[:, best_idx_sam2:best_idx_sam2+1],
    size=(1024, 1024),
    mode='bilinear',
    align_corners=False
)[0, 0].cpu().numpy()

print(f"[INFO] RepViT mask: min={mask_repvit_1024.min():.2f}, max={mask_repvit_1024.max():.2f}")
print(f"[INFO] SAM2 mask: min={mask_sam2_1024.min():.2f}, max={mask_sam2_1024.max():.2f}")

# 二值化 (logit > 0)
mask_repvit_binary = ((mask_repvit_1024 > 0) * 255).astype(np.uint8)
mask_sam2_binary = ((mask_sam2_1024 > 0) * 255).astype(np.uint8)

white_repvit = np.sum(mask_repvit_binary == 255)
white_sam2 = np.sum(mask_sam2_binary == 255)

print(f"[INFO] RepViT 白色像素: {white_repvit:,} ({white_repvit/mask_repvit_binary.size*100:.2f}%)")
print(f"[INFO] SAM2 白色像素: {white_sam2:,} ({white_sam2/mask_sam2_binary.size*100:.2f}%)")

# 转为OpenCV格式
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ========== 1. 纯黑白掩码对比 ==========
mask_comp_bw = np.hstack([mask_repvit_binary, mask_sam2_binary])
mask_comp_rgb = cv2.cvtColor(mask_comp_bw, cv2.COLOR_GRAY2BGR)

# 添加分割线和标注
cv2.line(mask_comp_rgb, (1024, 0), (1024, 1024), (0, 255, 0), 5)
cv2.putText(mask_comp_rgb, "RepViT+Adapter", (200, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)
cv2.putText(mask_comp_rgb, "SAM2 Native", (1300, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)
cv2.putText(mask_comp_rgb, f"White: {white_repvit:,} ({white_repvit/mask_repvit_binary.size*100:.1f}%)", 
            (100, 950), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
cv2.putText(mask_comp_rgb, f"White: {white_sam2:,} ({white_sam2/mask_sam2_binary.size*100:.1f}%)", 
            (1150, 950), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)

# 标记prompt点
cv2.circle(mask_comp_rgb, (point[0], point[1]), 15, (0, 0, 255), 3)
cv2.circle(mask_comp_rgb, (point[0], point[1]), 5, (255, 255, 255), -1)
cv2.circle(mask_comp_rgb, (1024 + point[0], point[1]), 15, (0, 0, 255), 3)
cv2.circle(mask_comp_rgb, (1024 + point[0], point[1]), 5, (255, 255, 255), -1)

cv2.imwrite('outputs/FINAL_CLEAN_MASKS.png', mask_comp_rgb)
print("[SAVED] outputs/FINAL_CLEAN_MASKS.png")

# ========== 2. 红绿叠加 ==========
overlay_repvit = img_cv.copy()
overlay_repvit[mask_repvit_binary == 255] = [0, 0, 255]  # 红色

overlay_sam2 = img_cv.copy()
overlay_sam2[mask_sam2_binary == 255] = [0, 255, 0]  # 绿色

overlay_comp = np.hstack([overlay_repvit, overlay_sam2])
cv2.line(overlay_comp, (1024, 0), (1024, 1024), (255, 255, 255), 5)
cv2.putText(overlay_comp, "RepViT (RED)", (250, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
cv2.putText(overlay_comp, "SAM2 (GREEN)", (1300, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)

cv2.imwrite('outputs/FINAL_CLEAN_OVERLAY.png', overlay_comp)
print("[SAVED] outputs/FINAL_CLEAN_OVERLAY.png")

# ========== 3. 热力图 ==========
# 将logits标准化到0-255
def normalize_to_uint8(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(arr, dtype=np.uint8)
    return normalized

heatmap_repvit_uint8 = normalize_to_uint8(mask_repvit_1024)
heatmap_sam2_uint8 = normalize_to_uint8(mask_sam2_1024)

heatmap_repvit_colored = cv2.applyColorMap(heatmap_repvit_uint8, cv2.COLORMAP_JET)
heatmap_sam2_colored = cv2.applyColorMap(heatmap_sam2_uint8, cv2.COLORMAP_JET)

heatmap_comp = np.hstack([heatmap_repvit_colored, heatmap_sam2_colored])
cv2.line(heatmap_comp, (1024, 0), (1024, 1024), (255, 255, 255), 5)
cv2.putText(heatmap_comp, "RepViT Heatmap", (220, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)
cv2.putText(heatmap_comp, "SAM2 Heatmap", (1270, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5, cv2.LINE_AA)

cv2.imwrite('outputs/FINAL_CLEAN_HEATMAP.png', heatmap_comp)
print("[SAVED] outputs/FINAL_CLEAN_HEATMAP.png")

print("\n" + "="*60)
print("所有可视化已完成！")
print("="*60)
print("1. FINAL_CLEAN_MASKS.png - 纯黑白掩码")
print("2. FINAL_CLEAN_OVERLAY.png - 彩色叠加")
print("3. FINAL_CLEAN_HEATMAP.png - 热力图")


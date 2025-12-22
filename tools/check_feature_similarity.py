#!/usr/bin/env python3
"""
计算 YOLOv8+Adapter 输出特征 与 NPZ P4_S16 特征 的相似度（余弦相似度），并打印分布统计。

使用方式：
  python tools/check_feature_similarity.py \
      --images_dir <dataset_dir> \
      --npz_dir outputs/features_v1_300 \
      --weights outputs/adapter_align_yolov8s_100ep.pth \
      --num 5 --device cuda
"""
import sys
from pathlib import Path
import argparse
import random
import json

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2


def load_model(weights_path: Path, device: torch.device):
    config = {'model_size': 's', 'pretrained': False, 'freeze_backbone': False}
    backbone = YOLOv8Backbone(config).to(device).eval()
    in_ch_s16 = backbone.get_feature_dims()[1]
    adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device).eval()

    ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['backbone'])
    adapter.load_state_dict(ckpt['adapter'])
    return backbone, adapter


def get_student_embedding(backbone, adapter, image_pil: Image.Image, device: torch.device) -> torch.Tensor:
    x = TF.to_tensor(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = backbone(x)
        emb = adapter(feats)
    return emb  # [1, 256, 64, 64]


def get_npz_embedding(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path))
    key = 'P4_S16' if 'P4_S16' in data.files else ('IMAGE_EMB_S16' if 'IMAGE_EMB_S16' in data.files else None)
    if key is None:
        raise KeyError(f"NPZ缺少P4_S16/IMAGE_EMB_S16: {npz_path}")
    arr = data[key]
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr  # [1, 256, 64, 64]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    cos = F.cosine_similarity(a, b).item()
    return float(cos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--npz_dir', type=str, default='outputs/features_v1_300')
    parser.add_argument('--weights', type=str, default='outputs/adapter_align_yolov8s_100ep.pth')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vs_sam2', action='store_true', help='同时计算学生与实时SAM2特征的相似度')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    images_dir = Path(args.images_dir)
    npz_dir = Path(args.npz_dir)
    weights = Path(args.weights)

    backbone, adapter = load_model(weights, device)

    # 可选加载SAM2用于实时特征
    sam2_model = None
    if args.vs_sam2:
        # stub 3rd-party if needed
        import types as _types
        for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
            if _mod not in sys.modules:
                stub_mod = _types.ModuleType(_mod)
                if _mod == 'loralib':
                    import torch as _torch
                    stub_mod.Linear = _torch.nn.Linear
                sys.modules[_mod] = stub_mod
        sam2_config_dir = _ROOT / 'vfmkd' / 'sam2' / 'sam2' / 'configs'
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
            sam2_model = build_sam2(
                config_file='sam2.1/sam2.1_hiera_b+.yaml',
                ckpt_path='weights/sam2.1_hiera_base_plus.pt',
                device=str(device)
            )
        sam2_model.eval()
        sam2_model.sam_mask_decoder.use_high_res_features = False

    # 采样文件
    all_images = sorted(list(images_dir.glob('*.jpg')))[:300]
    if len(all_images) == 0:
        print('[ERR] 未找到jpg图像')
        return
    sample_images = random.sample(all_images, min(args.num, len(all_images)))

    print(f"[INFO] 使用权重: {weights}")
    print(f"[INFO] 评估图像数: {len(sample_images)}\n")

    results = []
    for img_path in sample_images:
        image = Image.open(img_path).convert('RGB').resize((1024, 1024))
        stem = img_path.stem
        npz_path = npz_dir / f"{stem}_features.npz"
        if not npz_path.exists():
            print(f"[WARN] 缺少NPZ，跳过: {npz_path}")
            continue

        student = get_student_embedding(backbone, adapter, image, device)
        teacher_np = get_npz_embedding(npz_path)
        teacher = torch.from_numpy(teacher_np).to(device).float()

        with torch.no_grad():
            cos = cosine_similarity(student, teacher)
            s_mean, s_std = student.mean().item(), student.std().item()
            t_mean, t_std = teacher.mean().item(), teacher.std().item()

        results.append((img_path.name, cos, s_mean, s_std, t_mean, t_std))
        print(f"{img_path.name}: cos={cos:.4f} | s_mean={s_mean:.4f} s_std={s_std:.4f} | t_mean={t_mean:.4f} t_std={t_std:.4f}")

        if sam2_model is not None:
            with torch.no_grad():
                img_np = np.array(image).astype(np.float32) / 255.0
                img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
                out = sam2_model.image_encoder(img_t)
                rt = out['backbone_fpn'][2]
                cos_rt = cosine_similarity(student, rt)
                print(f"  vs SAM2: cos_rt={cos_rt:.4f} | rt_mean={rt.mean().item():.4f} rt_std={rt.std().item():.4f}")

    if results:
        cos_vals = [r[1] for r in results]
        print("\n[SUMMARY]")
        print(f"avg_cos={np.mean(cos_vals):.4f} | min={np.min(cos_vals):.4f} | max={np.max(cos_vals):.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
检查训练NPZ特征与实际SAM2分割特征的相似度
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear
        sys.modules[_mod] = stub_mod

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 设备: {device}\n")

# ========== 1. 加载SAM2模型 ==========
print("="*60)
print("1. 加载SAM2模型")
print("="*60)

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

# 初始化SAM2官方transform
sam2_transforms = SAM2Transforms(
    resolution=sam2_model.image_size,
    mask_threshold=0.0,
    max_hole_area=0.0,
    max_sprinkle_area=0.0
)
print("[OK] SAM2模型加载完成\n")

# ========== 2. 随机选择测试图片 ==========
print("="*60)
print("2. 随机选择测试图片")
print("="*60)

npz_dir = Path('outputs/features_v1_300')
all_npz = sorted(list(npz_dir.glob('*.npz')))
test_npz = random.sample(all_npz, min(5, len(all_npz)))

print(f"[INFO] 从{len(all_npz)}个NPZ文件中随机选择{len(test_npz)}个进行对比")
for i, npz_path in enumerate(test_npz, 1):
    print(f"  {i}. {npz_path.name}")
print()

# ========== 3. 对比特征 ==========
print("="*60)
print("3. 对比NPZ特征 vs 实际SAM2特征")
print("="*60)

image_dir = Path(r'C:\AiBuild\paper\detect\EdgeSAM-master\datasets\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0')

similarities = []

for idx, npz_path in enumerate(test_npz, 1):
    print(f"\n{'='*60}")
    print(f"测试 {idx}/{len(test_npz)}: {npz_path.stem}")
    print(f"{'='*60}")
    
    # 加载NPZ特征
    npz_data = np.load(npz_path)
    npz_feature = npz_data['P4_S16']  # (256, 64, 64)
    
    if npz_feature.ndim == 3:
        npz_feature = npz_feature[None, ...]  # (1, 256, 64, 64)
    
    npz_feature_torch = torch.from_numpy(npz_feature).float().to(device)
    
    print(f"[NPZ] P4_S16 shape: {npz_feature_torch.shape}")
    print(f"[NPZ] mean={npz_feature_torch.mean().item():.4f}, std={npz_feature_torch.std().item():.4f}")
    print(f"[NPZ] min={npz_feature_torch.min().item():.4f}, max={npz_feature_torch.max().item():.4f}")
    
    # 加载对应图片 (去掉_features后缀)
    image_id = npz_path.stem.replace('_features', '')
    image_path = image_dir / f"{image_id}.jpg"
    
    if not image_path.exists():
        print(f"[WARN] 图片不存在: {image_path}")
        continue
    
    # 用SAM2实时生成特征
    image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
    image_np = np.array(image_pil)
    
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        backbone_out = sam2_model.image_encoder(image_tensor)
        sam2_feature = backbone_out['backbone_fpn'][2]  # P4: (1, 256, 64, 64)
    
    print(f"[SAM2] P4 shape: {sam2_feature.shape}")
    print(f"[SAM2] mean={sam2_feature.mean().item():.4f}, std={sam2_feature.std().item():.4f}")
    print(f"[SAM2] min={sam2_feature.min().item():.4f}, max={sam2_feature.max().item():.4f}")
    
    # 计算相似度
    # 1. 余弦相似度
    npz_flat = npz_feature_torch.flatten()
    sam2_flat = sam2_feature.flatten()
    cos_sim = F.cosine_similarity(npz_flat.unsqueeze(0), sam2_flat.unsqueeze(0)).item()
    
    # 2. L2距离
    l2_dist = torch.norm(npz_feature_torch - sam2_feature, p=2).item()
    
    # 3. 均方误差
    mse = F.mse_loss(npz_feature_torch, sam2_feature).item()
    
    # 4. 平均绝对误差
    mae = torch.mean(torch.abs(npz_feature_torch - sam2_feature)).item()
    
    print(f"\n[对比结果]")
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  L2距离: {l2_dist:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    similarities.append({
        'image_id': image_id,
        'cos_sim': cos_sim,
        'l2_dist': l2_dist,
        'mse': mse,
        'mae': mae
    })

# ========== 4. 统计总结 ==========
print(f"\n{'='*60}")
print("4. 总体统计")
print(f"{'='*60}")

if similarities:
    avg_cos_sim = np.mean([s['cos_sim'] for s in similarities])
    avg_l2 = np.mean([s['l2_dist'] for s in similarities])
    avg_mse = np.mean([s['mse'] for s in similarities])
    avg_mae = np.mean([s['mae'] for s in similarities])
    
    print(f"平均余弦相似度: {avg_cos_sim:.6f}")
    print(f"平均L2距离: {avg_l2:.4f}")
    print(f"平均MSE: {avg_mse:.6f}")
    print(f"平均MAE: {avg_mae:.6f}")
    
    print(f"\n{'='*60}")
    print("结论:")
    print(f"{'='*60}")
    if avg_cos_sim > 0.99:
        print("✅ 特征高度一致 (余弦相似度 > 0.99)")
    elif avg_cos_sim > 0.95:
        print("⚠️ 特征基本一致 (余弦相似度 > 0.95)")
    elif avg_cos_sim > 0.90:
        print("⚠️ 特征存在一定差异 (余弦相似度 > 0.90)")
    else:
        print("❌ 特征差异较大 (余弦相似度 < 0.90)")
    
    print(f"\n说明:")
    print(f"- 训练时使用的NPZ特征与实际分割时SAM2生成的特征相似度为 {avg_cos_sim:.4f}")
    print(f"- 如果相似度较低，说明训练特征提取可能存在问题")
    print(f"- 如果相似度很高，说明YOLOv8+Adapter性能问题不在特征提取环节")

print(f"\n{'='*60}")
print("完成！")
print(f"{'='*60}")


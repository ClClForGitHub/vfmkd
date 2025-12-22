import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

import sys
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# 允许可选加载SAM2以做实时教师对比
_SAM2_ROOT = _ROOT / 'vfmkd' / 'sam2'
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter
try:
    from vfmkd.models.backbones.repvit_backbone import RepViT
    from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter
except Exception:
    RepViT = None

# 抑制第三方包的已知弃用告警（不影响功能）
warnings.filterwarnings(
    'ignore',
    message=r'Importing from timm\.models\.layers is deprecated',
    category=FutureWarning,
)


class NPZFeatureDataset(Dataset):
    """读取离线保存的 SAM2 embeddings（IMAGE_EMB_S16），并返回图像路径+teacher特征。"""

    def __init__(self, npz_dir: str, images_dir: str, image_size: int = 1024):
        self.npz_files = sorted([str(p) for p in Path(npz_dir).glob("*_features.npz")])
        self.images_dir = Path(images_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        import numpy as np
        npz_path = Path(self.npz_files[idx])
        # 使用mmap减少I/O抖动
        data = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        image_id = str(data['image_id'])
        # teacher embeddings: (1,256,64,64) 或 (256,64,64)
        # 注意：Teacher特征是用SAM2官方预处理（含ImageNet标准化）提取的
        # 兼容键名：优先使用 P4_S16，其次 IMAGE_EMB_S16
        if 'P4_S16' in data.files:
            teacher = data['P4_S16']
        elif 'IMAGE_EMB_S16' in data.files:
            teacher = data['IMAGE_EMB_S16']
        else:
            # 尝试其他可能键名
            for k in ['image_embedding', 'IMAGE_EMB', 'P4']:
                if k in data.files:
                    teacher = data[k]
                    break
            else:
                raise KeyError(f"Teacher feature not found in NPZ: expected keys ['P4_S16','IMAGE_EMB_S16'] in {npz_path}")
        # 标准化形状为 [256,64,64]（后续DataLoader堆叠 -> [B,256,64,64]）
        if teacher.ndim == 4 and teacher.shape[0] == 1:
            teacher = teacher[0]
        elif teacher.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected teacher shape: {teacher.shape} in {npz_path}")
        teacher = torch.from_numpy(np.asarray(teacher)).float()

        # YOLO学生模型预处理：仅Resize + /255（无ImageNet标准化）
        # 这与Teacher预处理不同，特征空间对齐由Adapter完成
        img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert('RGB').resize((self.image_size, self.image_size))
        x = TF.to_tensor(img)  # 自动 /255.0 转为 [0, 1]
        return x, teacher, image_id


@torch.no_grad()
def forward_backbone(backbone: YOLOv8Backbone, x: torch.Tensor) -> list:
    return backbone(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_dir', type=str, required=True, help='离线 teacher 特征目录')
    ap.add_argument('--images_dir', type=str, required=True, help='对应图片目录（文件名与NPZ image_id一致）')
    ap.add_argument('--yolov8_weight', type=str, default='')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='outputs/adapter_align.pth')
    ap.add_argument('--save_every', type=int, default=5, help='每隔多少个epoch保存一次checkpoint')
    ap.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    ap.add_argument('--fp16', action='store_true', help='使用混合精度加速')
    ap.add_argument('--unfreeze_backbone', action='store_true', help='解冻backbone参与训练')
    ap.add_argument('--no_external_weights', action='store_true', help='不加载外部YOLO权重，从随机初始化开始(仅yolov8)')
    ap.add_argument('--model_size', type=str, default='s', help='YOLOv8模型尺寸(n/s/m/l/x)')
    ap.add_argument('--backbone', type=str, default='yolov8', choices=['yolov8','repvit'], help='选择训练的backbone')
    # 诊断开关：在线对比NPZ Teacher与实时SAM2 Teacher
    ap.add_argument('--check_rt_sam2', action='store_true', help='开启：训练中定期计算实时SAM2特征，与NPZ特征做分布/相似度对比')
    ap.add_argument('--check_interval', type=int, default=50, help='开启实时对比时，每隔多少个step执行一次')
    ap.add_argument('--loss', type=str, default='mse', choices=['mse', 'cos', 'mix'], help='特征对齐损失')
    ap.add_argument('--mix_alpha', type=float, default=0.7, help='mix损失中MSE权重，余弦为(1-alpha)')
    args = ap.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    ds = NPZFeatureDataset(args.npz_dir, args.images_dir)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # 模型：冻结backbone，仅训1x1通道对齐适配器
    if args.backbone == 'yolov8':
        backbone_cfg = {
            'model_size': args.model_size,
            'pretrained': False,
            'external_weight_path': None if args.no_external_weights else (args.yolov8_weight if args.yolov8_weight else None),
            'freeze_backbone': not args.unfreeze_backbone,
        }
        backbone = YOLOv8Backbone(backbone_cfg).to(device)
    else:
        assert RepViT is not None, 'RepViT backbone not available'
        # RepViT 输出经过其内部neck已对齐到256通道
        backbone = RepViT('m1', img_size=1024, fuse=False, freeze=not args.unfreeze_backbone, load_from=None).to(device)
    if args.unfreeze_backbone:
        backbone.train()
        for p in backbone.parameters():
            p.requires_grad = True
    else:
        backbone.eval()
    use_adapter = (args.backbone == 'yolov8')
    adapter = None
    if use_adapter:
        in_ch_s16 = backbone.get_feature_dims()[1]
        adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device)
    else:
        # RepViT 对齐适配器（不改变通道，仅做LN+3x3+LN）
        adapter = RepViTAlignAdapter().to(device)

    params = []
    if adapter is not None:
        params += list(adapter.parameters())
    if args.unfreeze_backbone:
        params += [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)

    def cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # 计算每个样本的余弦相似度，按通道与空间展平
        b = student.shape[0]
        s = student.view(b, -1)
        t = teacher.view(b, -1)
        s = torch.nn.functional.normalize(s, dim=1)
        t = torch.nn.functional.normalize(t, dim=1)
        cos = (s * t).sum(dim=1).mean()
        return 1.0 - cos

    def mse_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(student, teacher)

    # 性能选项
    torch.backends.cudnn.benchmark = True

    # 可选：初始化SAM2用于实时教师对比
    sam2_model = None
    if args.check_rt_sam2:
        try:
            # 某些环境下需要stub第三方模块
            import types as _types
            for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
                if _mod not in sys.modules:
                    stub_mod = _types.ModuleType(_mod)
                    if _mod == 'loralib':
                        import torch as _torch
                        stub_mod.Linear = _torch.nn.Linear
                    sys.modules[_mod] = stub_mod

            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2

            sam2_config_dir = _SAM2_ROOT / 'sam2' / 'configs'
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
                sam2_model = build_sam2(
                    config_file='sam2.1/sam2.1_hiera_b+.yaml',
                    ckpt_path='weights/sam2.1_hiera_base_plus.pt',
                    device=str(device)
                )
            sam2_model.eval()
            # 为防接口不匹配，这里禁用高分支特征
            sam2_model.sam_mask_decoder.use_high_res_features = False
            print('[Diag] 实时SAM2教师对比已启用')
        except Exception as e:
            print(f'[Diag] 初始化SAM2失败，关闭实时对比: {e}')
            sam2_model = None

    for epoch in range(1, args.epochs + 1):
        if adapter is not None:
            adapter.train()
        total_loss = 0.0
        step_count = 0  # 重置每epoch的步数计数
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for x, teacher, image_id in pbar:
            x = x.to(device)
            teacher = teacher.to(device)
            # 兜底：若因旧数据导致 [B,1,256,64,64]，去掉多余维度
            if teacher.dim() == 5 and teacher.size(1) == 1:
                teacher = teacher[:, 0]
            # 前向
            if args.fp16 and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    feats = forward_backbone(backbone, x)
                    student = adapter(feats)
                    if student.shape != teacher.shape:
                        student = torch.nn.functional.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
                    if args.loss == 'mse':
                        loss = mse_loss(student, teacher)
                    elif args.loss == 'cos':
                        loss = cosine_loss(student, teacher)
                    else:  # mix
                        loss = args.mix_alpha * mse_loss(student, teacher) + (1.0 - args.mix_alpha) * cosine_loss(student, teacher)
            else:
                feats = forward_backbone(backbone, x)
                student = adapter(feats)
                if student.shape != teacher.shape:
                    student = torch.nn.functional.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
                if args.loss == 'mse':
                    loss = mse_loss(student, teacher)
                elif args.loss == 'cos':
                    loss = cosine_loss(student, teacher)
                else:
                    loss = args.mix_alpha * mse_loss(student, teacher) + (1.0 - args.mix_alpha) * cosine_loss(student, teacher)
            
            # 记录特征分布统计（每10个batch记录一次）
            step_count += 1
            if step_count % 10 == 0:
                with torch.no_grad():
                    s_mean = student.mean().item()
                    s_std = student.std().item()
                    s_min = student.min().item()
                    s_max = student.max().item()
                    t_mean = teacher.mean().item()
                    t_std = teacher.std().item()
                    t_min = teacher.min().item()
                    t_max = teacher.max().item()
                    pbar.set_postfix({
                        "loss": f"{loss.item():.5f}",
                        "s_mean": f"{s_mean:.3f}",
                        "s_std": f"{s_std:.3f}",
                        "t_mean": f"{t_mean:.3f}",
                        "t_std": f"{t_std:.3f}",
                        "s_range": f"[{s_min:.2f},{s_max:.2f}]",
                        "t_range": f"[{t_min:.2f},{t_max:.2f}]"
                    })

            # 可选：实时SAM2教师对比（每check_interval步执行一次，仅取batch中的第一个样本以控时）
            if sam2_model is not None and (step_count % max(1, args.check_interval) == 0):
                try:
                    with torch.no_grad():
                        # 取batch第一个样本，恢复为(1,3,1024,1024)的/255输入
                        img_1024 = x[0:1]  # 已是[0,1]
                        # SAM2 image_encoder 期望[0,1]即可（与我们已验证的流程一致）
                        out = sam2_model.image_encoder(img_1024)
                        rt_feat = out['backbone_fpn'][2]  # 64x64, 256c
                        # 统计
                        rt_mean = rt_feat.mean().item()
                        rt_std = rt_feat.std().item()
                        # 取对应的teacher样本（第一个）
                        t0 = teacher[0:1]
                        # 余弦相似度（展平、归一化）
                        s_rt = torch.nn.functional.normalize(rt_feat.flatten(1), dim=1)
                        s_t0 = torch.nn.functional.normalize(t0.flatten(1), dim=1)
                        cos_sim = (s_rt * s_t0).sum(dim=1).item()
                        print(f"[Diag][step {step_count}] image_id={image_id[0]} rt_mean={rt_mean:.3f} rt_std={rt_std:.3f} | npz_mean={t0.mean().item():.3f} npz_std={t0.std().item():.3f} | cos={cos_sim:.4f}")
                except Exception as e:
                    print(f"[Diag] 实时对比失败: {e}")

            optimizer.zero_grad()
            # 统一反传调用
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch}: loss={avg:.6f}")

        # 按间隔保存checkpoint（包含adapter+backbone）
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt_path = Path(args.out).with_suffix("")
            ckpt_path = f"{ckpt_path}_epoch{epoch}.pth"
            Path(os.path.dirname(ckpt_path) or '.').mkdir(parents=True, exist_ok=True)
            torch.save({
                'adapter': adapter.state_dict(),
                'backbone': backbone.state_dict(),
                'epoch': epoch,
                'avg_loss': avg,
            }, ckpt_path)
            print(f"[CKPT] saved: {ckpt_path}")

    Path(os.path.dirname(args.out) or '.').mkdir(parents=True, exist_ok=True)
    torch.save({'adapter': adapter.state_dict(), 'backbone': backbone.state_dict()}, args.out)
    print(f"Saved adapter+backbone weights to {args.out}")


if __name__ == '__main__':
    main()



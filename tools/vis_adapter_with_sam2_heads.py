import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import sys
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'vfmkd' / 'sam2') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))
# EdgeSAM repo path for SAM1.0 heads
_ESAM = (_ROOT.parent / 'EdgeSAM-master')
if _ESAM.exists() and str(_ESAM) not in sys.path:
    sys.path.insert(0, str(_ESAM))
_ESAM_NEST = (_ROOT.parent / 'EdgeSAM-master' / 'EdgeSAM-master')
if _ESAM_NEST.exists() and str(_ESAM_NEST) not in sys.path:
    sys.path.insert(0, str(_ESAM_NEST))

# Stub heavy deps to import heads-only
import types as _types
for _mod in ['mmdet', 'mmcv', 'mmengine', 'loralib']:
    if _mod not in sys.modules:
        stub_mod = _types.ModuleType(_mod)
        # 为loralib添加Linear占位类
        if _mod == 'loralib':
            stub_mod.Linear = torch.nn.Linear  # type: ignore
        sys.modules[_mod] = stub_mod

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_image_adapter import Sam2ImageAdapter
try:
    from vfmkd.models.backbones.repvit_backbone import RepViT, RepViTBackbone
    from vfmkd.models.heads.repvit_align_adapter import RepViTAlignAdapter
except Exception:
    RepViT = None

# SAM2 heads
from sam2.modeling.sam.prompt_encoder import PromptEncoder as PromptEncoderSAM2
from sam2.modeling.sam.mask_decoder import MaskDecoder as MaskDecoderSAM2
from sam2.modeling.sam.transformer import TwoWayTransformer as TwoWayTransformerSAM2
try:
    from edge_sam.modeling.prompt_encoder import PromptEncoder as PromptEncoderEdge
    from edge_sam.modeling.mask_decoder import MaskDecoder as MaskDecoderEdge
    from edge_sam.modeling.transformer import TwoWayTransformer as TwoWayTransformerEdge
except Exception as e:
    print(f"[WARN] import edge_sam heads failed: {e}")
    PromptEncoderEdge = None
    # Fallback: try dynamic load from file paths
    try:
        import importlib.util
        def _dyn_load(name, file_path):
            spec = importlib.util.spec_from_file_location(name, file_path)
            if spec is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        for base in [_ESAM_NEST, _ESAM]:
            if base and (base / 'edge_sam' / 'modeling').exists():
                pe_mod = _dyn_load('edge_pe', str(base / 'edge_sam' / 'modeling' / 'prompt_encoder.py'))
                md_mod = _dyn_load('edge_md', str(base / 'edge_sam' / 'modeling' / 'mask_decoder.py'))
                tr_mod = _dyn_load('edge_tr', str(base / 'edge_sam' / 'modeling' / 'transformer.py'))
                if pe_mod and md_mod and tr_mod:
                    PromptEncoderEdge = getattr(pe_mod, 'PromptEncoder', None)
                    MaskDecoderEdge = getattr(md_mod, 'MaskDecoder', None)
                    TwoWayTransformerEdge = getattr(tr_mod, 'TwoWayTransformer', None)
                    break
    except Exception as e2:
        print(f"[WARN] dynamic load edge_sam heads failed: {e2}")
        PromptEncoderEdge = None


def load_image_rgb(path: str, size: int = 1024) -> torch.Tensor:
    img = Image.open(path).convert('RGB').resize((size, size))
    x = TF.to_tensor(img).unsqueeze(0)
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--yolov8_weight', type=str, default='')
    ap.add_argument('--adapter_weight', type=str, default='outputs/adapter_align.pth')
    ap.add_argument('--backbone_weight', type=str, default='', help='训练得到的backbone权重(与适配器同轮次)')
    ap.add_argument('--backbone', type=str, default='yolov8', choices=['yolov8','repvit'], help='选择推理backbone')
    ap.add_argument('--heads', type=str, default='sam2', choices=['sam2','edgesam'], help='选择提示/掩码头(SAM2或EdgeSAM)')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_stem', type=str, default='outputs/adapter_sam2_cand')
    ap.add_argument('--point', type=float, nargs=2, default=[512, 512])
    ap.add_argument('--label', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # Backbone
    if args.backbone == 'yolov8':
        bb_model = YOLOv8Backbone({
            'model_size': 'n',
            'pretrained': False,
            'external_weight_path': args.yolov8_weight if args.yolov8_weight else None,
        }).to(device).eval()
    else:
        assert RepViT is not None, 'RepViT not available'
        import inspect
        print(f"[DEBUG] RepViT symbol = {RepViT}, module = {getattr(RepViT, '__module__', 'unknown')}")
        # Prefer wrapper to avoid edge instantiation issues
        bb_model = RepViTBackbone({'arch': 'm1', 'img_size': 1024, 'fuse': False, 'freeze': False, 'load_from': None})
        assert isinstance(bb_model, torch.nn.Module), 'RepViTBackbone build failed'
        bb_model = bb_model.to(device).eval()

    # 可选加载训练得到的backbone权重（非严格部分加载）。
    # 支持两种来源：
    # 1) --backbone_weight 指定的文件（优先）
    # 2) 若未提供，则尝试从 --adapter_weight 中读取 'backbone' 键
    def _try_load_backbone(path: str) -> bool:
        try:
            ckptb = torch.load(path, map_location='cpu')
            state_b = ckptb.get('backbone', ckptb)
            missing, unexpected = bb_model.load_state_dict(state_b, strict=False)
            print(f"[INFO] backbone state load from {path}: missing={len(missing)}, unexpected={len(unexpected)}")
            return True
        except Exception as e:
            print(f"[WARN] load backbone weight failed from {path}: {e}")
            return False

    loaded_bb = False
    if args.backbone_weight and Path(args.backbone_weight).exists():
        loaded_bb = _try_load_backbone(args.backbone_weight)
    if (not loaded_bb) and args.adapter_weight and Path(args.adapter_weight).exists():
        # 尝试从适配器权重中读取 backbone（同一文件保存的情况）
        loaded_bb = _try_load_backbone(args.adapter_weight)

    if args.backbone == 'yolov8':
        in_ch_s16 = bb_model.get_feature_dims()[1]
        adapter = Sam2ImageAdapter(in_channels_s16=in_ch_s16).to(device).eval()
    else:
        adapter = RepViTAlignAdapter().to(device).eval()
    if args.adapter_weight and Path(args.adapter_weight).exists():
        try:
            ckpt = torch.load(args.adapter_weight, map_location='cpu')
            state = ckpt.get('adapter', ckpt)
            adapter.load_state_dict(state, strict=False)
            print(f"[INFO] adapter state loaded from {args.adapter_weight}")
        except Exception as e:
            print(f"[WARN] load adapter weight failed from {args.adapter_weight}: {e}")

    # Prompt/Mask heads
    if args.heads == 'sam2':
        prompt_encoder = PromptEncoderSAM2(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ).to(device).eval()
        mask_decoder = MaskDecoderSAM2(
            num_multimask_outputs=3,
            transformer=TwoWayTransformerSAM2(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=False,
            iou_prediction_use_sigmoid=True,
        ).to(device).eval()
    else:
        assert PromptEncoderEdge is not None, 'EdgeSAM heads not available'
        prompt_encoder = PromptEncoderEdge(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ).to(device).eval()
        mask_decoder = MaskDecoderEdge(
            num_multimask_outputs=3,
            transformer=TwoWayTransformerEdge(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=False,
        ).to(device).eval()

    # Data
    x = load_image_rgb(args.image, 1024).to(device)
    print(f"[DEBUG] backbone type: {type(bb_model)}")
    feats = bb_model(x)
    img_emb = adapter(feats)

    pts = torch.tensor([[args.point]], device=device, dtype=torch.float32)
    lbl = torch.tensor([[args.label]], device=device, dtype=torch.int64)
    sparse, dense = prompt_encoder(points=(pts, lbl), boxes=None, masks=None)

    # SAM2 mask decoding（低分辨率256x256 logits）
    if args.heads == 'sam2':
        low_res, ious, _, _ = mask_decoder(
            image_embeddings=img_emb,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
        )
    else:
        low_res, ious = mask_decoder(
            image_embeddings=img_emb,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            num_multimask_outputs=3,
        )

    Path(os.path.dirname(args.out_stem) or '.').mkdir(parents=True, exist_ok=True)
    # 保存三张候选（保持原生低分辨率256x256，不上采样）
    for i in range(low_res.shape[1]):
        p = torch.sigmoid(low_res[0, i])  # (256,256)
        m = (p > float(args.threshold)).float()
        Image.fromarray((m.cpu().numpy() * 255).astype('uint8')).save(f"{args.out_stem}_cand{i}.png")
    print(f"saved cands to {args.out_stem}_cand[0-2].png, ious={ious[0].tolist()}")


if __name__ == '__main__':
    main()





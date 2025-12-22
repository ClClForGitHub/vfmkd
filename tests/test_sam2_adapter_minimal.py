import torch

from vfmkd.models.backbones.yolov8_backbone import YOLOv8Backbone
from vfmkd.models.heads.sam2_adapter_head import SAM2SegAdapterHead


@torch.no_grad()
def test_minimal_point_infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造 YOLOv8 骨干
    backbone = YOLOv8Backbone({
        "model_size": "n",
        "pretrained": False,
    }).to(device)

    # 构造 SAM2 适配头；in_channels 需与 YOLOv8Backbone 对齐
    in_dims = backbone.get_feature_dims()
    head = SAM2SegAdapterHead({
        "image_size": 1024,
        "hidden_dim": 256,
        "in_channels": in_dims,
        "backbone_strides": backbone.get_feature_strides(),
        "use_high_res_features": True,
        "num_multimask_outputs": 3,
    }).to(device)

    # 伪造一张输入图
    x = torch.randn(1, 3, 1024, 1024, device=device)
    feats = backbone(x)  # [s8, s16, s32]

    # 单点提示（前景点，位于图像中心附近），坐标在 1024 坐标系
    pts = torch.tensor([[[512.0, 512.0]]], device=device)
    lbl = torch.tensor([[1]], device=device)

    out = head(
        feats,
        point_coords=pts,
        point_labels=lbl,
        boxes=None,
        mask_inputs=None,
        num_multimask_outputs=1,
    )

    low_res = out["low_res_logits"]
    assert low_res.shape[-2:] == (256, 256), f"unexpected mask size: {low_res.shape}"

    print("low_res_logits:", tuple(low_res.shape))


if __name__ == "__main__":
    test_minimal_point_infer()



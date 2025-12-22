# VFMKD Distillation Experiment for YOLOv8s on COCO.
# This configuration file defines the distillation-only training phase.

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# Overwrite the default scope to include our custom vfmkd modules
default_scope = 'vfmkd'

# Define the distillation model using our custom VFMKDYOLODistiller
model = dict(
    type='VFMKDYOLODistiller',
    # Student components
    backbone=dict(
        type='YOLOv8Backbone',
        model_size='s',
        # It's good practice to define init_cfg, e.g., for pretrained weights
        init_cfg=dict(type='Pretrained', checkpoint='weights/yolov8s.pt', prefix='model.0.')
    ),
    adapter=dict(
        type='Sam2ImageAdapter',
        in_channels_s16=256,  # YOLOv8s P4 out_channels
        out_channels=256
    ),
    # Teacher model configuration (placeholder)
    # The actual teacher features should be loaded from pre-computed files.
    # This is a dummy config for MMDetection's model builder.
    teacher_cfg=dict(
        type='mmcls.ResNet', # Using a placeholder from mmcls
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth')
    ),
    # Distillation loss
    distillation_loss=dict(
        type='mmdet.L1Loss',
        loss_weight=1.0
    )
)

# Since this is a distillation-only phase, we don't need evaluation hooks
# that expect bounding box predictions. We can disable them.
val_evaluator = None
test_evaluator = None

# We might also need a custom dataset that yields teacher features alongside images.
# For now, we assume the distiller's forward handles this.

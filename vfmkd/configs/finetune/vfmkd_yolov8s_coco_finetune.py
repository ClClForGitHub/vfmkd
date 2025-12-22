# VFMKD Fine-tuning Experiment for YOLOv8s on COCO.
# This config defines a standard detector for fine-tuning and mAP evaluation.
# It loads the backbone weights from a checkpoint produced by the distillation phase.

_base_ = [
    '../_base_/models/vfmkd_yolov8s.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# Overwrite the default scope to include our custom vfmkd modules
default_scope = 'vfmkd'

# Load the distilled backbone weights using init_cfg
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            # IMPORTANT: Change this path to your actual distilled checkpoint
            checkpoint='outputs/coco_yolov8_fgd_backbone_unfrozen/models/epoch_10.pth',
            # The prefix is crucial to map the saved weights correctly.
            # In our distiller, the backbone is saved under `backbone`.
            prefix='backbone.'
        )
    ),
    # The neck and head will be randomly initialized.
)

# You can adjust learning rate, epochs, etc., for the fine-tuning stage.
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25, # Fine-tune for fewer epochs
        by_epoch=True,
        milestones=[15, 22],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=25, val_interval=1)

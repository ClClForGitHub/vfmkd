# Base model configuration for our custom VFMKD YOLOv8s detector.
# This config defines the standard detector architecture using our refactored modules.

model = dict(
    type='YOLOV3',  # We use YOLOV3 as a placeholder detector type that fits the structure.
    backbone=dict(
        type='YOLOv8Backbone',
        model_size='s'),
    neck=dict(
        type='YOLOv8PAFPN',
        model_size='s'),
    bbox_head=dict(
        type='YOLOV8Head',  # Placeholder, a proper MMDetection head is needed.
        num_classes=80,
        in_channels=[128, 256, 512], # Corresponds to YOLOv8s neck out_channels
        featmap_strides=[8, 16, 32]),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

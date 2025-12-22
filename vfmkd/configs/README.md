# VFMKD MMDetection Workflow

This directory contains the configuration files for running VFMKD experiments using the MMDetection framework. Our custom modules are defined within the `vfmkd` source directory and are automatically discovered by MMDetection's runner thanks to the `setup.py` installation.

## Prerequisites

1.  **Installation**: Ensure you have installed MMDetection and its dependencies.
2.  **Install VFMKD**: Install our project in editable mode from the root directory:
    ```bash
    pip install -e .
    ```
3.  **Data**: Prepare the COCO dataset according to MMDetection's guidelines and ensure the `data_root` in `_base_/datasets/coco_detection.py` points to the correct location.

## How to Run Experiments

All experiments should be launched from the project's root directory using MMDetection's standard tools.

### 1. Distillation Phase

This phase trains the student backbone using feature distillation.

**Command:**
```bash
# Single GPU training
python tools/train.py vfmkd/configs/distillation/vfmkd_yolov8s_coco_distill.py

# Multi-GPU training
./tools/dist_train.sh vfmkd/configs/distillation/vfmkd_yolov8s_coco_distill.py <NUM_GPUS>
```

Checkpoints and logs will be saved to the `work_dirs/vfmkd_yolov8s_coco_distill/` directory.

### 2. Fine-tuning & Evaluation Phase

This phase loads the distilled backbone weights, attaches a neck and head, and fine-tunes the full detector for mAP evaluation.

**Before you run:**
-   Open `vfmkd/configs/finetune/vfmkd_yolov8s_coco_finetune.py`.
-   Modify the `checkpoint` path in `model.backbone.init_cfg` to point to the actual checkpoint file generated during the distillation phase (e.g., `work_dirs/vfmkd_yolov8s_coco_distill/epoch_50.pth`).

**Training Command:**
```bash
# Single GPU training
python tools/train.py vfmkd/configs/finetune/vfmkd_yolov8s_coco_finetune.py

# Multi-GPU training
./tools/dist_train.sh vfmkd/configs/finetune/vfmkd_yolov8s_coco_finetune.py <NUM_GPUS>
```

### 3. Testing (mAP Evaluation)

To evaluate the mAP of a fine-tuned checkpoint without retraining:

**Command:**
```bash
# Single GPU testing
python tools/test.py vfmkd/configs/finetune/vfmkd_yolov8s_coco_finetune.py <PATH_TO_FINETUNED_CHECKPOINT> --show

# Multi-GPU testing
./tools/dist_test.sh vfmkd/configs/finetune/vfmkd_yolov8s_coco_finetune.py <PATH_TO_FINETUNED_CHECKPOINT> <NUM_GPUS> --out results.pkl
```

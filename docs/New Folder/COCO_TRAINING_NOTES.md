# COCO Training Notes (2025-11-17)

## Summary
- Dataset root: `/home/team/zouzhiyuan/dataset/COCO2017`
- Training script: `tools/core/train/train_coco_mmdet_lego.py`
- Current runs  
  - **GPU5**: unfrozen backbone (`work_dirs/coco_gpu5_unfrozen`)  
  - **GPU6**: frozen backbone, unfreeze at epoch 50 (`work_dirs/coco_gpu6_frozen`)

## Issues Encountered & Fixes

### 1. Custom Module Imports Failed
- Files: `vfmkd/models/heads/__init__.py`, `vfmkd/models/heads/detection/__init__.py`, `vfmkd/models/heads/sam2_image_adapter.py`
- Problem: legacy code pasted without indentation caused `IndentationError`.
- Fix: reindent all `try/except` blocks. Verified `vfmkd.models` registry imports cleanly.

### 2. COCO Dataset Path & Collate Mismatch
- Problem: original config still pointed to `data/coco` and used `keep_ratio=True`, so `yolov5_collate` received variable-sized tensors → `stack` runtime error.
- Fixes:
  - Overrode `cfg.data_root` to the actual dataset path and rewired `data_prefix` to `images/train2017/`, `images/val2017/`.
  - Forced `keep_ratio=False` with `Resize(scale=(640, 640))` for train/test pipelines.
  - Registered `yolov5_collate` via `FUNCTIONS.register_module(...)` and set `cfg.train_dataloader.collate_fn = dict(type='yolov5_collate')`.

### 3. Backbone/Neck/Head Channel Alignment
- Problem: manually specified channels `[128,256,512,...]` no longer matched the actual CSPDarknet outputs, leading to mismatched convolution weights inside the neck/head.
- Fix: derive channel dims dynamically from `model_size='s'`:
  - Backbone outputs `[c2, c2, c3, c4] = [64, 64, 128, 256]`.
  - Neck takes those dims directly.
  - Head uses base `[256, 512, 1024]` with `widen_factor=width_mult` so MMYOLO scales internally.

### 4. Dataloader AssertionError (`data_samples` not dict)
- Trigger: `yolov5_collate` not registered, so MMEngine attempted to pretty-print config and failed on `<function ...>`.
- Fix already covered in §2 via registry + dict-based collate config.

## Commands Used

```bash
# Unfrozen backbone on GPU5
CUDA_VISIBLE_DEVICES=5 python tools/core/train/train_coco_mmdet_lego.py \
  --distilled-backbone outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth \
  --work-dir ./work_dirs/coco_gpu5_unfrozen --bs 16

# Frozen backbone (unfreeze at epoch 50) on GPU6
CUDA_VISIBLE_DEVICES=6 python tools/core/train/train_coco_mmdet_lego.py \
  --distilled-backbone outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth \
  --work-dir ./work_dirs/coco_gpu6_frozen --bs 16 \
  --freeze-backbone --unfreeze-at-epoch 50
```

_Both commands are currently running via `nohup`, see `work_dirs/coco_gpu*/pid.txt`._

## Checkpoints & Metrics
- Checkpoints saved under each `work_dir` (default MMEngine `epoch_X.pth` naming).  
- Evaluation (`val_interval=1`) will start producing mAP once the first epoch completes.  
- To inspect on-the-fly results:
  - Logs: `tail -f work_dirs/coco_gpu5_unfrozen/train.log`
  - Logs: `tail -f work_dirs/coco_gpu6_frozen/train.log`

## Next Steps
- Monitor loss + mAP trends (especially once backbone unfreezes on GPU6).
- Consider enabling advanced YOLO augmentations (mosaic/mixup) once baseline stabilizes.
- Optionally add eval-only run using the saved checkpoints.



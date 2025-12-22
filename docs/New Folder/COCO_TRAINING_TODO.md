# COCO训练验证和测试 TODO List

## 已完成 ✅

1. ✅ **验证MMDetection组件注册状态**
   - 所有自定义组件（YOLOv8Backbone, YOLOv8PAFPN, YOLOv8DetectHead, Sam2ImageAdapter, VFMKDYOLODistiller）已成功注册

2. ✅ **检查训练脚本train_coco_mmdet_lego.py的checkpoint加载逻辑**
   - 脚本存在且逻辑完整
   - 支持从checkpoint提取backbone权重
   - 使用临时文件保存权重供MMDetection加载

3. ✅ **验证蒸馏模型checkpoint格式和键名**
   - Checkpoint包含：`backbone`, `edge_adapter`, `edge_head`, `feature_adapter`, `optimizer`, `scaler`, `epoch`, `config`
   - `backbone`键存在且包含162个参数
   - 发现`_orig_mod.`前缀问题（torch.compile结果）

4. ✅ **检查并修复训练脚本中的配置文件路径**
   - 所有配置文件存在：
     - `vfmkd/configs/_base_/datasets/coco_detection.py`
     - `vfmkd/configs/_base_/schedules/schedule_1x.py`
     - `vfmkd/configs/_base_/default_runtime.py`

5. ✅ **修复checkpoint中_orig_mod前缀的处理**
   - 已在训练脚本中添加前缀移除逻辑

## 进行中 🔄

6. 🔄 **验证模型构建配置（YOLOv8Backbone, YOLOv8PAFPN与MMYOLO组件的兼容性）**
   - 需要测试模型是否能正确构建
   - 需要验证通道数匹配
   - ⚠️ 注意：当前环境存在matplotlib GLIBCXX问题，但不影响训练脚本运行

## 待完成 📋

7. ✅ **创建测试脚本验证checkpoint加载和模型构建**
   - ✅ 已创建 `test_checkpoint_loading.py`
   - ✅ 已验证checkpoint格式和_orig_mod前缀处理
   - ⚠️ 由于环境matplotlib问题，完整测试需要在实际训练中验证

8. 📋 **测试冻结backbone的训练流程（--freeze-backbone参数）**
   - 测试backbone冻结功能
   - 验证optimizer参数组设置
   - 验证frozen_stages设置

9. 📋 **测试不冻结backbone的训练流程（默认或--unfreeze-at-epoch参数）**
   - 测试默认训练（backbone不冻结）
   - 测试UnfreezeBackboneHook功能
   - 验证在指定epoch解冻backbone

10. 📋 **运行完整的COCO训练测试（冻结backbone）**
    - 使用小数据集或少量epoch进行测试
    - 验证训练流程完整性
    - 检查日志和输出

11. 📋 **运行完整的COCO训练测试（不冻结backbone）**
    - 使用小数据集或少量epoch进行测试
    - 验证训练流程完整性
    - 检查日志和输出

## 已知问题 ⚠️

1. **Checkpoint格式问题**
   - Checkpoint中的backbone参数包含`_orig_mod.`前缀（torch.compile结果）
   - ✅ 已修复：在训练脚本中添加了前缀移除逻辑

2. **环境依赖问题**
   - matplotlib和Pillow存在GLIBCXX版本问题
   - ✅ 已解决：
     - 降级matplotlib到3.6.3
     - 降级Pillow到9.5.0
     - 在训练脚本中设置LD_LIBRARY_PATH使用conda环境的libstdc++

## 下一步行动

### 立即执行：
1. 创建测试脚本验证checkpoint加载和模型构建
2. 测试模型前向传播是否正常

### 后续测试：
1. 使用小数据集测试冻结backbone训练
2. 使用小数据集测试不冻结backbone训练
3. 验证UnfreezeBackboneHook在指定epoch解冻backbone

## 测试命令示例

### 冻结backbone训练：
```bash
python tools/core/train/train_coco_mmdet_lego.py \
    --distilled-backbone outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth \
    --freeze-backbone \
    --work-dir ./work_dirs/coco_finetune_frozen \
    --bs 16
```

### 不冻结backbone训练：
```bash
python tools/core/train/train_coco_mmdet_lego.py \
    --distilled-backbone outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth \
    --work-dir ./work_dirs/coco_finetune_unfrozen \
    --bs 16
```

### 先冻结后解冻训练：
```bash
python tools/core/train/train_coco_mmdet_lego.py \
    --distilled-backbone outputs/distill_single_test_MSE/20251115_175318_yolov8_no_edge_boost_full_train_with_diagnostics/models/epoch_4_model.pth \
    --freeze-backbone \
    --unfreeze-at-epoch 50 \
    --work-dir ./work_dirs/coco_finetune_unfreeze_50 \
    --bs 16
```


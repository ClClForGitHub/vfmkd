# Box Prompts 处理逻辑修复说明

## 问题分析

### 1. 数据接收问题

**原始问题**：
- 在 `_parse_npz_data_fast` 函数中，`box_prompts_flag` 的逻辑有问题：
  - 第556行直接设置 `box_prompts_flag = 1`，假设只要有 `box_prompts_xyxy` 和 `box_prompts_masks_256` 就有框
  - 但实际上应该根据 `box_prompts_count` 来判断是否有框
  - 第632行会覆盖 `box_prompts_flag`，但如果 NPZ 中没有这个字段，会用之前的值（可能是错误的）

**修复方案**：
- 优先使用 NPZ 中的 `box_prompts_count`，如果没有则从 `box_prompts_xyxy.shape[0]` 推断
- 根据 `box_prompts_count > 0` 来设置 `box_prompts_flag`
- 在最后验证 NPZ 中的 `box_prompts_flag` 是否与 `box_prompts_count` 一致，如果不一致则以 `count` 为准

### 2. 数据处理问题

**原始问题**：
- 在 `_forward_mask_head` 方法中，提取 box 和 mask 时：
  - 第2040行和2042行都假设每个样本的第一个框（索引0）存在
  - 但如果 `box_prompts_xyxy[b_idx]` 是 `[0, 4]` 的空 tensor，访问 `[0]` 会出错
  - 同样，第2046行和2048行也有同样的问题

**修复方案**：
- 在提取 box 和 mask 之前，先检查 tensor 是否为空（`numel() > 0`）
- 检查 tensor 的第一维是否大于0（`shape[0] > 0`）
- 如果为空或没有数据，跳过该样本（虽然理论上不应该发生，因为 `cnt == 1` 已经检查过）

### 3. 数据格式理解

**关键点**：
- `box_prompts_xyxy` 和 `box_prompts_masks_256` 在 `_custom_collate_fn` 中被保持为**列表**格式
- 每个列表元素是一个 tensor：
  - `box_prompts_xyxy[i]`: `[N, 4]` 的 tensor（N 是该样本的框数量，通常是 0 或 1）
  - `box_prompts_masks_256[i]`: `[N, 256, 256]` 的 tensor（N 是该样本的 mask 数量，通常是 0 或 1）
- `box_prompts_count` 是一个列表或 tensor，每个元素是该样本的框数量（通常是 0 或 1）

## 修复后的逻辑流程

### 1. 数据解析阶段（`_parse_npz_data_fast`）

```python
# 优先使用预处理好的 box_prompts_*
if "box_prompts_xyxy" in npz_data and "box_prompts_masks_256" in npz_data:
    # 读取数据
    box_prompts_xyxy = torch.from_numpy(npz_data["box_prompts_xyxy"])
    box_prompts_masks_256 = torch.from_numpy(npz_data["box_prompts_masks_256"])
    
    # 优先使用 NPZ 中的 box_prompts_count，否则从形状推断
    if "box_prompts_count" in npz_data:
        box_prompts_count = int(npz_data["box_prompts_count"])
    else:
        box_prompts_count = box_prompts_xyxy.shape[0] if box_prompts_xyxy.numel() > 0 else 0
    
    # 根据 count 判断是否有框
    box_prompts_flag = 1 if box_prompts_count > 0 else 0
    
    # 最后验证 NPZ 中的 flag 是否与 count 一致
    if "box_prompts_flag" in npz_data:
        npz_flag = int(npz_data["box_prompts_flag"])
        if (npz_flag == 1 and box_prompts_count == 0) or (npz_flag == 0 and box_prompts_count > 0):
            # 不一致，以 count 为准（已在上面设置）
            pass
        else:
            box_prompts_flag = npz_flag
```

### 2. 训练循环阶段（`_forward_mask_head`）

```python
# 解析 counts
if isinstance(box_prompts_count, torch.Tensor):
    counts = box_prompts_count.tolist()
elif isinstance(box_prompts_count, list):
    counts = [int(c) for c in box_prompts_count]
else:
    counts = [int(box_prompts_count)]

# 只处理有框的样本（cnt == 1）
for b_idx in range(batch_size):
    cnt = int(counts[b_idx]) if b_idx < len(counts) else 0
    if cnt != 1:
        continue  # 跳过没有框的样本
    
    # 提取 box（先检查是否为空）
    box_tensor = box_prompts_xyxy[b_idx]  # 列表格式
    if isinstance(box_tensor, torch.Tensor) and box_tensor.numel() > 0:
        if box_tensor.shape[0] > 0:
            curr_box = box_tensor[0, :].to(device, dtype=torch.float32, non_blocking=True)
        else:
            continue  # 空 tensor，跳过
    
    # 提取 mask（先检查是否为空）
    mask_tensor = box_prompts_masks[b_idx]  # 列表格式
    if isinstance(mask_tensor, torch.Tensor) and mask_tensor.numel() > 0:
        if mask_tensor.shape[0] > 0:
            curr_gt = mask_tensor[0].to(device, dtype=torch.float32, non_blocking=True)
        else:
            continue  # 空 tensor，跳过
```

## 实际 NPZ 数据结构

**重要**：实际 NPZ 文件中**没有** `box_prompts_xyxy`、`box_prompts_masks_256`、`box_prompts_count` 这三个键！

实际 NPZ 文件中包含的字段：
- `bboxes`: `[N, 4]` **XYWH格式** `[x, y, w, h]`，原图坐标系
- `masks`: object array，每个元素是 `[H_orig, W_orig]` 的原图尺寸掩码
- `image_shape`: `[H, W, C]` 或 `[H, W]`，原图尺寸
- `has_bbox`: bool，是否有框
- `num_bboxes`: int，框的数量

## 转换逻辑

代码会从 `bboxes` 和 `masks` 转换得到：
1. `box_prompts_xyxy`: 从 `bboxes` 的 XYWH 格式转换为 XYXY 格式
2. `box_prompts_masks_256`: 从 `masks` 的原图尺寸缩放到 256x256
3. `box_prompts_count`: 从 `bboxes.shape[0]` 或 `num_bboxes` 获取

## 验证要点

### 1. 框位置（box_prompts_xyxy）
- ✅ 能准确接收：从 NPZ 的 `bboxes` 字段读取（XYWH格式）
- ✅ 格式转换：正确从 XYWH `[x, y, w, h]` 转换为 XYXY `[x0, y0, x1, y1]`
- ✅ 坐标系：原图坐标系，不缩放（SAM2内部会处理）
- ✅ 处理正确：在 `_forward_mask_head` 中正确提取并转换坐标

### 2. 掩码图（box_prompts_masks_256）
- ✅ 能准确接收：从 NPZ 的 `masks` 字段读取（原图尺寸）
- ✅ 格式转换：从原图尺寸缩放到 `[N, 256, 256]` float32 格式
- ✅ 处理正确：在 `_forward_mask_head` 中正确提取并处理维度

### 3. 是否有框（box_prompts_count / box_prompts_flag）
- ✅ 能准确接收：从 NPZ 的 `bboxes.shape[0]` 或 `num_bboxes` 获取
- ✅ 逻辑正确：根据 `box_prompts_count > 0` 设置 `box_prompts_flag`
- ✅ 验证一致：检查 NPZ 中的 `has_bbox` 是否与 `count` 一致

## 修复的文件

1. `tools/core/exper/train_distill_single_test.py`
   - `_parse_npz_data_fast` 函数（第543-633行）
     - 修复 `box_prompts_flag` 逻辑
     - **修复 XYWH -> XYXY 格式转换**（关键修复！）
   - `TarShardNPZDataset.__getitem__` 方法（第1110-1139行）
     - 修复 `box_prompts_flag` 逻辑
     - **修复 XYWH -> XYXY 格式转换**（关键修复！）
   - `_forward_mask_head` 方法（第2038-2059行）
     - 添加空值检查，正确处理列表格式

## 关键修复：XYWH -> XYXY 格式转换

**问题**：NPZ 中的 `bboxes` 是 XYWH 格式 `[x, y, w, h]`，但代码注释错误地说是 XYXY 格式，导致直接使用会出错。

**修复**：
```python
# 转换XYWH -> XYXY: [x, y, w, h] -> [x0, y0, x1, y1]
bboxes_xyxy = np.zeros_like(bboxes_np)
bboxes_xyxy[:, 0] = bboxes_np[:, 0]  # x0 = x
bboxes_xyxy[:, 1] = bboxes_np[:, 1]  # y0 = y
bboxes_xyxy[:, 2] = bboxes_np[:, 0] + bboxes_np[:, 2]  # x1 = x + w
bboxes_xyxy[:, 3] = bboxes_np[:, 1] + bboxes_np[:, 3]  # y1 = y + h
```

## 测试建议

1. **测试有框的样本**：验证 `box_prompts_count == 1` 时能正确提取 box 和 mask
2. **测试无框的样本**：验证 `box_prompts_count == 0` 时能正确跳过
3. **测试数据一致性**：验证 NPZ 中的 `box_prompts_flag` 与 `box_prompts_count` 一致
4. **测试边界情况**：验证空 tensor、None 值等边界情况的处理


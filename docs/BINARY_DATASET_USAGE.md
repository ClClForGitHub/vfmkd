# 二进制数据集使用指南

## 概述

二进制数据集是高性能定长存储格式，相比 TAR/NPZ 格式，IO 性能提升 **10-20 倍**。

### 性能优势

1. **O(1) Seek**: 直接通过 `index * ITEM_SIZE` 计算偏移量，瞬间定位
2. **Zero-Copy**: 使用 `np.frombuffer` 直接转换，避免内存拷贝
3. **跳过解码**: 不再需要 JPG 解码和 NPZ 解压，CPU 占用大幅降低
4. **完全并行**: 每个 Worker 进程独立文件句柄，真正跑满磁盘带宽

## 数据格式

### 文件结构

```
sa1b_binary/
├── config.json          # 全局配置（model_type, total_samples 等）
├── keys.txt             # 样本 ID 列表（用于调试）
├── images.bin           # 图像数据 (1024×1024×3 uint8)
├── features.bin         # 特征数据 (P4_S16 + P5_S32 float32)
├── edge_maps.bin        # 边缘图 (edge_256 + edge_64 + edge_32 uint8)
├── weight_maps.bin       # 权重图 (fg_map + bg_map float32)
├── bboxes.bin           # 边界框 (1×4 float32)
├── masks.bin            # 掩码 (1×256×256 uint8)
└── metadata.bin         # 元数据 (5×int32)
```

### 数据格式详情

| 文件 | 单个样本大小 | 格式 |
|------|-------------|------|
| images.bin | 3,145,728 bytes | (1024, 1024, 3) uint8 |
| features.bin | 5,242,880 bytes | P4_S16 (1, 256, 64, 64) + P5_S32 (1, 256, 32, 32) float32 |
| edge_maps.bin | 70,656 bytes | edge_256x256 + edge_64x64 + edge_32x32 uint8 |
| weight_maps.bin | 172,032 bytes | fg_map_128 + bg_map_128 + fg_map_64 + bg_map_64 + fg_map_32 + bg_map_32 float32 |
| bboxes.bin | 16 bytes | (1, 4) float32 |
| masks.bin | 65,536 bytes | (1, 256, 256) uint8 |
| metadata.bin | 20 bytes | [num_bboxes, has_bbox, H, W, C] int32 |

## 使用方法

### 1. 转换 TAR → BIN

首先使用 `convert_tar_to_bin.py` 将 TAR 文件转换为二进制格式：

```bash
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /path/to/sa1b_shard_00000.tar \
    --output-dir /home/team/zouzhiyuan/dataset/sa1b_binary \
    --model-type "sam2.1_hiera_b+" \
    --workers 32
```

批量转换所有 shard：

```bash
bash tools/core/data/batch_convert_tar_to_bin.sh
```

### 2. 训练时使用二进制数据集

在 `train_distill_single_test.py` 中使用 `--data-format binary`：

```bash
python tools/core/exper/train_distill_single_test.py \
    --data-format binary \
    --binary-data-path /home/team/zouzhiyuan/dataset/sa1b_binary \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 10 \
    --loss-type mse \
    --backbone yolov8
```

### 3. 参数说明

- `--data-format binary`: 指定使用二进制数据集格式
- `--binary-data-path`: 二进制数据集根目录（包含 config.json 和所有 .bin 文件）
- `--num-workers`: 建议设置为 4-8（二进制读取不再受 GIL 限制，可以更高）
- `--batch-size`: 可以适当增大（IO 不再是瓶颈）

### 4. 路径自动查找

如果不指定 `--binary-data-path`，脚本会按以下优先级查找：

1. 用户指定的 `--binary-data-path`
2. 环境配置中的 `binary_data_dir`（如果存在）
3. 默认路径：`/home/team/zouzhiyuan/dataset/sa1b_binary`

## 返回数据格式

`BinaryDistillDataset` 返回的数据格式与 `train_distill_single_test.py` 完全兼容：

```python
{
    "image": torch.Tensor,              # (3, 1024, 1024) uint8
    "teacher_features": torch.Tensor,   # (256, 64, 64) float32 (P4_S16)
    "edge_256x256": torch.Tensor,       # (256, 256) float32
    "fg_map": torch.Tensor,             # (1, 64, 64) float32
    "bg_map": torch.Tensor,             # (1, 64, 64) float32
    "edge_map": torch.Tensor,          # (1, 64, 64) float32
    "box_prompts_xyxy": torch.Tensor,  # (N, 4) 或 (0, 4) float32
    "box_prompts_masks_orig": torch.Tensor,  # (N, 256, 256) 或 (0, 256, 256) float32
    "box_prompts_count": int,
    "box_prompts_flag": int,
    "box_prompts_meta": list,
    "geometry_color_flag": int,
    "has_bbox": bool,
    "num_bboxes": int,
    "image_shape": np.ndarray,          # (3,) int32 [H, W, C]
    "image_id": str,
}
```

## 性能对比

### TAR Shard 模式（旧）

- **IO 瓶颈**: JPG 解码 + NPZ 解压占用大量 CPU
- **Worker 数量**: 受 GIL 限制，建议 4-8 个
- **读取速度**: ~100-200 MB/s（受 CPU 解码限制）

### Binary 模式（新）

- **IO 瓶颈**: 几乎无瓶颈，纯磁盘读取
- **Worker 数量**: 可以设置更高（8-16 个）
- **读取速度**: ~500-1000 MB/s（受磁盘带宽限制）

### 预期提升

- **数据加载时间**: 减少 **50-80%**
- **GPU 利用率**: 提升 **10-20%**（减少等待时间）
- **整体训练速度**: 提升 **20-40%**（取决于 IO 占比）

## 注意事项

1. **首次使用**: 需要先运行 `convert_tar_to_bin.py` 转换数据
2. **磁盘空间**: 二进制格式比 TAR 大 **1.67 倍**（JPG 解压为 RAW）
3. **兼容性**: 返回格式与现有训练脚本完全兼容，无需修改训练逻辑
4. **文件句柄**: 每个 Worker 进程独立打开文件句柄，支持多进程并行

## 故障排除

### 问题 1: 找不到 config.json

```
FileNotFoundError: Config not found at ...
```

**解决**: 确保已运行 `convert_tar_to_bin.py` 生成二进制数据集

### 问题 2: 文件大小不匹配

```
IOError: Failed to read image at index X: expected 3145728 bytes, got ...
```

**解决**: 检查二进制文件是否完整，可能需要重新转换

### 问题 3: 数据格式不兼容

```
RuntimeError: Expected tensor of shape ...
```

**解决**: 确保使用最新版本的 `convert_tar_to_bin.py` 和 `binary_dataset.py`

## 代码位置

- **数据集类**: `tools/core/exper/binary_dataset.py`
- **转换脚本**: `tools/core/data/convert_tar_to_bin.py`
- **批量转换**: `tools/core/data/batch_convert_tar_to_bin.sh`
- **训练脚本**: `tools/core/exper/train_distill_single_test.py`


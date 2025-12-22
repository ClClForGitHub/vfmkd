# 二进制数据集输出目录结构

## 📁 最终输出目录结构

### 方案 A：统一目录（推荐）

```
/home/team/zouzhiyuan/dataset/sa1b_binary/
├── config.json              # 全局配置（所有 shard 共享）
│   └── 包含: model_type, image_size, mask_size, total_samples, sample_sizes
│
├── keys.txt                 # 所有样本的 ID 列表（按 shard 顺序）
│   └── 格式: sa_000000\nsa_000001\n...（一行一个 ID）
│
├── images.bin               # 所有样本的图像数据（按 shard 顺序拼接）
│   └── 每个样本: 3,145,728 bytes (1024×1024×3 uint8)
│
├── features.bin             # 所有样本的特征数据（按 shard 顺序拼接）
│   └── 每个样本: 5,242,880 bytes (P4_S16 + P5_S32 float32)
│
├── edge_maps.bin            # 所有样本的边缘图（按 shard 顺序拼接）
│   └── 每个样本: 70,656 bytes (edge_256x256 + edge_64x64 + edge_32x32 uint8)
│
├── weight_maps.bin          # 所有样本的权重图（按 shard 顺序拼接）
│   └── 每个样本: 172,032 bytes (fg_map_* + bg_map_* float32)
│
├── bboxes.bin               # 所有样本的边界框（按 shard 顺序拼接）
│   └── 每个样本: 16 bytes ((1, 4) float32)
│
├── masks.bin                # 所有样本的掩码（按 shard 顺序拼接）
│   └── 每个样本: 65,536 bytes ((1, 256, 256) uint8)
│
└── metadata.bin             # 所有样本的元数据（按 shard 顺序拼接）
    └── 每个样本: 20 bytes (num_bboxes + has_bbox + image_shape int32)
```

**特点**：
- ✅ 所有数据在一个目录，便于管理
- ✅ 所有 shard 的数据按顺序拼接，索引连续
- ✅ 单个文件可能很大（100+ GB），但读取性能最优
- ✅ 适合顺序训练和随机访问

### 方案 B：按 Shard 分目录（备选）

```
/home/team/zouzhiyuan/dataset/sa1b_binary_shards/
├── shard_00000/
│   ├── config.json
│   ├── keys.txt
│   ├── images.bin
│   ├── features.bin
│   ├── edge_maps.bin
│   ├── weight_maps.bin
│   ├── bboxes.bin
│   ├── masks.bin
│   └── metadata.bin
├── shard_00001/
│   └── ... (同上)
└── ...
```

**特点**：
- ✅ 每个 shard 独立，便于并行处理
- ✅ 文件大小适中（每个 shard 约 8-9 GB）
- ✅ 可以按需加载特定 shard
- ⚠️ 需要额外的索引文件来映射全局样本 ID

## 📊 空间估算

### 输入数据
- **TAR 文件总数**: 110 个
- **TAR 文件总大小**: 547 GB
- **预计样本总数**: 110,000 个（假设每个 shard 1000 个样本）

### 输出数据
- **单个样本大小**: 8.3 MB
  - images: 3.0 MB
  - features: 5.0 MB
  - edge_maps: 69 KB
  - weight_maps: 168 KB
  - bboxes: 16 bytes
  - masks: 64 KB
  - metadata: 20 bytes

- **总输出大小**: 约 **913 GB** (0.89 TB)
  - 比 TAR 文件大 1.67 倍（因为 JPG 解压为 RAW）

### 磁盘空间需求
- **最小需求**: 1.5 TB（输出 913 GB + 临时空间）
- **推荐**: 2 TB 以上（预留空间）

## 💾 磁盘存储检查

### 当前磁盘状态
- **主分区** (`/dev/sda4`): 1.7T 容量，已用 179G，可用 **1.5T**，使用率 11%
- **TAR 文件位置**: `/home/team/zouzhiyuan/dataset/sa1b_tar_shards/` (547 GB)

### 推荐输出位置

**方案 1：统一目录（推荐）**
```
/home/team/zouzhiyuan/dataset/sa1b_binary/
```

**理由**：
- ✅ 与 TAR 文件在同一数据集目录下，便于管理
- ✅ 主分区有 1.5T 可用空间，足够存储 913 GB 输出
- ✅ 路径清晰，符合项目规范

**方案 2：项目目录下**
```
/home/team/zouzhiyuan/vfmkd/datasets/sa1b_binary/
```

**理由**：
- ✅ 与项目代码在一起，便于版本管理
- ⚠️ 但数据集通常不放在代码目录下

## 🚀 转换脚本使用

### 批量转换所有 Shard

```bash
# 创建输出目录
mkdir -p /home/team/zouzhiyuan/dataset/sa1b_binary

# 批量转换脚本（需要创建）
for i in {0..109}; do
    shard_num=$(printf "%05d" $i)
    echo "处理 shard_${shard_num}..."
    python tools/core/data/convert_tar_to_bin.py \
        --tar-path /home/team/zouzhiyuan/dataset/sa1b_tar_shards/sa1b_shard_${shard_num}.tar \
        --output-dir /home/team/zouzhiyuan/dataset/sa1b_binary \
        --model-type "sam2.1_hiera_b+" \
        --workers 32 \
        --quiet
done
```

### 单 Shard 测试

```bash
# 测试转换（只转换前 100 个样本）
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /home/team/zouzhiyuan/dataset/sa1b_tar_shards/sa1b_shard_00000.tar \
    --output-dir /home/team/zouzhiyuan/dataset/sa1b_binary_test \
    --max-samples 100 \
    --model-type "sam2.1_hiera_b+"
```

## 📋 文件格式说明

### config.json
```json
{
  "model_type": "sam2.1_hiera_b+",
  "image_size": 1024,
  "mask_size": 256,
  "total_samples": 110000,
  "version": "1.0",
  "description": "SA-1B dataset converted to fixed-length binary format",
  "sample_sizes": {
    "image_bytes": 3145728,
    "features_bytes": 5242880,
    "edge_maps_bytes": 70656,
    "weight_maps_bytes": 172032,
    "bboxes_bytes": 16,
    "masks_bytes": 65536,
    "metadata_bytes": 20
  },
  "interpolation_method": "cv2.INTER_AREA",
  "mask_binarization_threshold": 0.5
}
```

### keys.txt
```
sa_000000
sa_000001
sa_000002
...
```

### 二进制文件格式
- 所有 `.bin` 文件都是**定长记录**，按样本索引顺序存储
- 第 `i` 个样本的数据位于：`offset = i * SAMPLE_SIZE`
- 可以直接使用 `seek()` 和 `read()` 进行随机访问

## ✅ 最终推荐

**输出目录**: `/home/team/zouzhiyuan/dataset/sa1b_binary/`

**理由**：
1. ✅ 磁盘空间充足（1.5T 可用 > 913 GB 需求）
2. ✅ 与 TAR 文件在同一数据集目录，便于管理
3. ✅ 路径清晰，符合项目规范
4. ✅ 统一目录结构，所有 shard 数据连续存储，索引简单


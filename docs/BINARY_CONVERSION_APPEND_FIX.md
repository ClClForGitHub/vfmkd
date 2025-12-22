# 二进制转换追加模式修复

## 问题描述

**严重问题**：之前的 `convert_tar_to_bin.py` 脚本使用覆盖模式（`"wb"`），导致批量转换时，每个 shard 都会覆盖之前的数据，最终只保留了最后一个 shard 的数据。

### 问题原因

在 `convert_tar_to_bin.py` 中，所有输出文件都以 `"wb"`（write binary，覆盖模式）打开：

```python
files = {
    "images": open(output_dir / "images.bin", "wb"),  # ❌ 覆盖模式
    "features": open(output_dir / "features.bin", "wb"),
    # ...
}
f_keys = open(output_dir / "keys.txt", "w", encoding="utf-8")  # ❌ 覆盖模式
```

当批量转换脚本循环处理多个 shard 时：
1. `shard_00000.tar` → 写入数据到 `images.bin` 等文件
2. `shard_00001.tar` → **清空并重新创建** `images.bin` 等文件，覆盖了 shard_00000 的数据
3. ... 以此类推
4. 最终结果：只有最后一个 shard 的数据被保留

## 修复方案

### 1. 添加 `--append` 参数

在 `convert_tar_to_bin.py` 中添加了 `--append` 参数，支持追加模式：

```python
def convert_tar_to_bin(
    # ... 其他参数 ...
    append: bool = False,  # 新增参数
) -> Dict[str, int]:
    # 决定文件打开模式
    mode_bin = "ab" if append else "wb"  # 追加 vs 覆盖
    mode_txt = "a" if append else "w"
    
    # 打开文件（根据 append 参数决定模式）
    files = {
        "images": open(output_dir / "images.bin", mode_bin),  # ✅ 支持追加
        # ...
    }
    f_keys = open(output_dir / "keys.txt", mode_txt, encoding="utf-8")
```

### 2. 处理 `config.json` 累加逻辑

追加模式需要累加 `total_samples`：

```python
# 计算总样本数（追加模式需要累加）
total_samples = stats["success"]
if append and config_path.exists():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            old_config = json.load(f)
            # 累加总样本数
            if "total_samples" in old_config:
                old_total = int(old_config["total_samples"])
                total_samples = old_total + stats["success"]
                if verbose:
                    print(f"📊 累加样本数: {old_total} (已有) + {stats['success']} (本次) = {total_samples} (总计)")
    except Exception as e:
        if verbose:
            print(f"⚠️  警告: 读取旧 config.json 失败: {e}，将使用本次样本数")
```

### 3. 修改批量转换脚本

在 `batch_convert_tar_to_bin.sh` 中，从第二个 shard 开始使用 `--append`：

```bash
# 从第二个 shard 开始（i > 0），添加 --append 参数
if [ $i -gt 0 ]; then
    CMD_ARGS+=("--append")
    echo "  模式: 追加（追加到已有数据）"
else
    echo "  模式: 覆盖（创建新文件）"
fi
```

## 使用方法

### 单个 Shard 转换（覆盖模式）

```bash
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /path/to/sa1b_shard_00000.tar \
    --output-dir /home/team/zouzhiyuan/dataset/sa1b_binary \
    --model-type "sam2.1_hiera_b+" \
    --workers 32
```

### 单个 Shard 转换（追加模式）

```bash
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /path/to/sa1b_shard_00001.tar \
    --output-dir /home/team/zouzhiyuan/dataset/sa1b_binary \
    --model-type "sam2.1_hiera_b+" \
    --workers 32 \
    --append  # ✅ 追加模式
```

### 批量转换（自动处理）

使用批量转换脚本，会自动处理追加逻辑：

```bash
bash tools/core/data/batch_convert_tar_to_bin.sh
```

脚本会自动：
- 第一个 shard（shard_00000）：覆盖模式（创建新文件）
- 后续 shard（shard_00001 到 shard_00109）：追加模式（追加到已有文件）

## 验证修复

### 检查文件大小

转换完成后，检查输出文件大小：

```bash
ls -lh /home/team/zouzhiyuan/dataset/sa1b_binary/*.bin
```

预期大小（110 个 shard，每个约 1000 个样本）：
- `images.bin`: 约 330 GB
- `features.bin`: 约 550 GB
- `edge_maps.bin`: 约 7.6 GB
- `weight_maps.bin`: 约 18.5 GB
- `bboxes.bin`: 约 1.8 MB
- `masks.bin`: 约 7.0 GB
- `metadata.bin`: 约 2.2 MB

### 检查 config.json

```bash
cat /home/team/zouzhiyuan/dataset/sa1b_binary/config.json | grep total_samples
```

应该显示累加后的总样本数（例如：110,000）。

### 检查 keys.txt

```bash
wc -l /home/team/zouzhiyuan/dataset/sa1b_binary/keys.txt
```

应该显示所有 shard 的样本总数。

## 紧急修复步骤

如果之前已经运行了批量转换（导致数据被覆盖），需要：

1. **停止当前任务**（如果还在运行）
2. **删除不完整的数据**：
   ```bash
   rm -rf /home/team/zouzhiyuan/dataset/sa1b_binary
   ```
3. **重新运行批量转换**：
   ```bash
   bash tools/core/data/batch_convert_tar_to_bin.sh
   ```

## 修改的文件

1. **`tools/core/data/convert_tar_to_bin.py`**
   - 添加 `append` 参数到 `convert_tar_to_bin` 函数
   - 根据 `append` 参数决定文件打开模式
   - 实现 `config.json` 的累加逻辑
   - 在 `main` 函数中添加 `--append` 命令行参数

2. **`tools/core/data/batch_convert_tar_to_bin.sh`**
   - 从第二个 shard 开始自动添加 `--append` 参数
   - 显示当前使用的模式（覆盖/追加）

## 注意事项

1. **首次转换**：第一个 shard 必须使用覆盖模式（默认），创建新文件
2. **后续转换**：从第二个 shard 开始必须使用追加模式（`--append`）
3. **重新转换**：如果要重新转换，需要先删除输出目录，否则会追加到旧数据上
4. **config.json**：追加模式会自动累加 `total_samples`，确保最终值正确

## 测试验证

修复后，可以测试追加功能：

```bash
# 测试：转换第一个 shard
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /path/to/sa1b_shard_00000.tar \
    --output-dir /tmp/test_binary \
    --max-samples 10

# 检查文件大小
ls -lh /tmp/test_binary/*.bin

# 测试：追加第二个 shard
python tools/core/data/convert_tar_to_bin.py \
    --tar-path /path/to/sa1b_shard_00001.tar \
    --output-dir /tmp/test_binary \
    --max-samples 10 \
    --append

# 再次检查文件大小（应该翻倍）
ls -lh /tmp/test_binary/*.bin

# 检查 config.json（total_samples 应该累加）
cat /tmp/test_binary/config.json | grep total_samples
```


# SAM2 分割头部模块使用说明

本目录提供了SAM2分割功能的离线单例管理器，支持使用SAM2的Prompt Encoder和Mask Decoder生成分割掩码。

## 目录结构

```
tools/core/prompt/
├── builder.py          # SAM2模型构建函数
├── prompts.py          # 框提示提取函数
├── seg_head.py         # SAM2SegHead单例管理器
├── test_seg_head.py    # 测试脚本
└── README.md           # 本文件
```

## 模块说明

### 1. `seg_head.py` - SAM2SegHead

SAM2分割头部管理器（单例模式），提供离线初始化和掩码生成功能。

#### 特性

- **单例模式**：首次初始化后，后续调用复用同一实例，避免重复加载模型
- **离线初始化**：使用`build_sam2`加载完整SAM2模型，提取prompt_encoder和mask_decoder
- **权重保障**：确保prompt_encoder和mask_decoder的权重正确加载
- **无需Adapter**：外部保证特征已对齐为(B, 256, 64, 64)格式

#### 使用示例

```python
from tools.core.prompt.seg_head import SAM2SegHead
import torch

# 初始化（仅首次调用，后续复用）
head = SAM2SegHead.get_instance(device='cuda')

# 准备输入（外部保证已对齐）
student_features = torch.randn(1, 256, 64, 64).cuda()  # 已对齐的特征
box_tensor = torch.tensor([[100, 100, 500, 500]]).cuda()  # [x0, y0, x1, y1]

# 生成掩码
result = head.generate_mask(
    aligned_features=student_features,
    boxes=box_tensor,
    multimask_output=True
)

# 获取结果
best_mask = result['best_mask']  # (1, 1, 256, 256) 最佳掩码logits
best_iou = result['best_iou']     # (1, 1) 最佳IOU值
iou_predictions = result['iou_predictions']  # (1, 3) 所有IOU预测

# 上采样到原图尺寸
mask_1024 = head.upsample_mask(
    best_mask,
    target_size=(1024, 1024),
    return_binary=True
)  # (1, 1, 1024, 1024)
```

#### API 文档

##### `SAM2SegHead.get_instance(device='cuda')`

获取单例实例（懒加载模式）。

**参数：**
- `device` (str): 设备字符串，默认'cuda'

**返回：**
- `SAM2SegHead`: 单例实例

##### `generate_mask()`

生成分割掩码。

**参数：**
- `aligned_features` (torch.Tensor): 已对齐的特征，形状(B, 256, 64, 64)
- `boxes` (torch.Tensor, optional): 框提示，形状(B, 4)或(N, 4)，格式[x0, y0, x1, y1]，坐标在1024×1024坐标系
- `points` (tuple, optional): 点提示元组(points_coords, point_labels)，或None
- `masks` (torch.Tensor, optional): 掩码提示，形状(B, 1, H, W)，或None
- `multimask_output` (bool): 是否输出多个掩码候选，默认True

**返回：**
- `dict` 包含：
  - `masks`: (B, M, 256, 256) 低分辨率掩码logits，M=3 if multimask_output else 1
  - `iou_predictions`: (B, M) IOU预测值
  - `best_mask`: (B, 1, 256, 256) 最佳掩码（IOU最高）
  - `best_iou`: (B, 1) 最佳IOU值
  - `best_idx`: (B,) 最佳掩码索引

##### `upsample_mask()`

上采样掩码到目标尺寸。

**参数：**
- `mask_logits` (torch.Tensor): 低分辨率掩码logits，形状(B, 1, 256, 256)或(B, M, 256, 256)
- `target_size` (tuple): 目标尺寸，默认(1024, 1024)
- `mode` (str): 插值模式，'bilinear'或'nearest'，默认'bilinear'
- `threshold` (float): 二值化阈值（logits空间），默认0.0
- `return_binary` (bool): 是否返回二值掩码，默认False

**返回：**
- `torch.Tensor`: 上采样后的掩码，形状(B, C, H, W)

### 2. `prompts.py` - 框提示提取

提供从JSON标注文件中提取框提示的功能。

#### 使用示例

```python
from tools.core.prompt.prompts import bbox_from_json
from pathlib import Path

# 从JSON提取框
json_path = Path('path/to/annotation.json')
boxes_tensor, box_xyxy = bbox_from_json(
    json_path, 
    target_size=1024,
    strategy='best_single',  # 或 'first_n', 'all'
    num_boxes=1
)

print(f"提取到 {boxes_tensor.shape[0]} 个框")
# boxes_tensor: (N, 4) tensor
# box_xyxy: (x0, y0, x1, y1) tuple
```

#### 函数说明

##### `score_instance_by_bbox()`

对实例进行打分，用于选择最合适的框。

**策略：**
- 越接近target_ratio（默认0.10）越好
- 惩罚极端长宽比与条带
- 偏好较高填充度

##### `bbox_from_json()`

从JSON标注文件提取框提示。

**参数：**
- `json_path` (Path): JSON标注文件路径
- `target_size` (int): 目标尺寸，默认1024
- `pad_ratio` (float): 框的padding比例，默认0.05
- `strategy` (str): 选择策略
  - `'best_single'`: 选择最合适的单个实例（默认）
  - `'first_n'`: 选择前N个实例（按score排序）
  - `'all'`: 选择所有实例
- `num_boxes` (int): 当strategy='first_n'时，选择的框数量

**返回：**
- `boxes_tensor`: (N, 4) float32 tensor，坐标已映射到[0, target_size-1]
- `box_xyxy`: (x0, y0, x1, y1) int元组（第一个框的坐标）

### 3. `builder.py` - 模型构建

提供SAM2模型和学生模型的构建函数。

#### `build_sam2()`

构建SAM2模型（关闭高分支特征）。

**参数：**
- `device` (torch.device): 设备

**返回：**
- `SAM2Base`: 完整的SAM2模型（eval模式）

## 测试脚本

### `test_seg_head.py`

使用NPZ特征作为"学生特征"进行测试的脚本。

#### 使用方法

```bash
python tools/core/prompt/test_seg_head.py \
    --npz_dir /path/to/npz/features \
    --images_dir /path/to/images \
    --output_dir outputs/seg_head_test \
    --num_samples 5 \
    --device cuda \
    --box_strategy best_single
```

#### 参数说明

- `--npz_dir`: NPZ特征文件目录
- `--images_dir`: 图像目录
- `--output_dir`: 输出目录，默认`outputs/seg_head_test`
- `--num_samples`: 测试样本数量，默认5
- `--device`: 设备，默认'cuda'
- `--box_strategy`: 选框策略，可选'best_single'、'first_n'、'all'，默认'best_single'
- `--num_boxes`: 当strategy=first_n时，选择的框数量，默认1

#### 输出

脚本会在输出目录生成对比可视化图片，每张图片包含：
1. 原图 + 选框
2. 学生模型生成的掩码
3. 教师模型生成的掩码
4. 叠加对比（显示一致性）

## 工作流程

### 1. 特征对齐（外部完成）

在调用`SAM2SegHead.generate_mask()`之前，需要确保特征已经对齐到SAM2的特征空间：

- **输入要求**：特征必须是 `(B, 256, 64, 64)` 格式
- **对齐方式**：使用Adapter（如`RepViTAlignAdapter`或`Sam2ImageAdapter`）将学生模型特征对齐到SAM2特征空间

### 2. 框提示提取

使用`bbox_from_json()`从JSON标注文件中提取框提示：

```python
boxes_tensor, _ = bbox_from_json(json_path, target_size=1024)
```

### 3. 掩码生成

使用`SAM2SegHead.generate_mask()`生成掩码：

```python
result = head.generate_mask(
    aligned_features=student_features,
    boxes=boxes_tensor,
    multimask_output=True
)
```

### 4. 后处理

使用`upsample_mask()`上采样到原图尺寸：

```python
mask_1024 = head.upsample_mask(
    result['best_mask'],
    target_size=(1024, 1024),
    return_binary=True
)
```

## 注意事项

1. **权重加载**：`build_sam2`使用`strict=True`加载权重，确保所有组件（包括prompt_encoder和mask_decoder）的权重都正确加载。

2. **特征对齐**：本模块不负责特征对齐，需要外部保证输入特征已经是`(B, 256, 64, 64)`格式。

3. **设备管理**：所有输入张量会自动迁移到正确的设备，但建议在输入前确保张量在正确的设备上。

4. **单例模式**：`SAM2SegHead`是单例模式，首次调用`get_instance()`会初始化，后续调用返回同一实例。如需重置，调用`reset_instance()`。

5. **坐标系**：框提示的坐标应该在1024×1024坐标系下，`bbox_from_json()`会自动处理映射。

## 完整示例

```python
import torch
from pathlib import Path
from tools.core.prompt.seg_head import SAM2SegHead
from tools.core.prompt.prompts import bbox_from_json

# 1. 初始化（仅一次）
head = SAM2SegHead.get_instance(device='cuda')

# 2. 准备输入
# 假设已有对齐后的特征（外部保证）
student_features = torch.randn(1, 256, 64, 64).cuda()

# 3. 提取框提示
json_path = Path('path/to/annotation.json')
boxes_tensor, _ = bbox_from_json(json_path, target_size=1024)
boxes_tensor = boxes_tensor.to('cuda')

# 4. 生成掩码
result = head.generate_mask(
    aligned_features=student_features,
    boxes=boxes_tensor,
    multimask_output=True
)

# 5. 上采样
mask_1024 = head.upsample_mask(
    result['best_mask'],
    target_size=(1024, 1024),
    return_binary=True
)

print(f"最佳IOU: {result['best_iou'][0, 0].item():.4f}")
print(f"掩码shape: {mask_1024.shape}")
```

## 故障排除

### 问题：权重加载失败

**解决方案**：确保`weights/sam2.1_hiera_base_plus.pt`文件存在且完整。

### 问题：特征形状不匹配

**错误信息**：`Expected input shape (B, 256, 64, 64)`

**解决方案**：确保输入特征已经对齐到`(B, 256, 64, 64)`格式。

### 问题：设备不匹配

**解决方案**：确保所有输入张量在正确的设备上，或让模块自动处理设备迁移。

## 更新日志

- 2024-XX-XX: 初始版本
  - 实现SAM2SegHead单例管理器
  - 实现框提示提取功能
  - 添加测试脚本


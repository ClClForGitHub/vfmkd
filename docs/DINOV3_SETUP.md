# DinoV3 集成指南

本指南介绍如何将 Meta AI 的 DinoV3 模型 vendor 到本地并集成到 VFMKD 框架中。

## 📋 概述

DinoV3 是 Meta AI 发布的自监督视觉基础模型，具有强大的特征表示能力。我们将其 vendor 到 `vfmkd/dinov3` 目录，类似于 SAM2 的集成方式。

## 🔧 安装步骤

### 1. Vendor DinoV3 代码

运行 vendor 脚本将 DinoV3 官方仓库克隆到本地：

```bash
cd /home/team/zouzhiyuan/vfmkd
python tools/vendor_dinov3.py
```

这将：
- 克隆 DinoV3 仓库到 `vfmkd/dinov3/` 目录
- 如果目录已存在，会尝试更新（使用 `--force` 强制重新克隆）

可选参数：
- `--repo-url`: 指定仓库 URL（默认: https://github.com/facebookresearch/dinov3.git）
- `--target-dir`: 指定目标目录（默认: vfmkd/dinov3）
- `--force`: 强制重新克隆（删除现有目录）

### 2. 下载预训练权重

下载 DinoV3 的预训练权重：

```bash
python tools/download_dinov3_weights.py --model base
```

支持的模型类型：
- `small`: DinoV3-S (ViT-S/14, 22M 参数)
- `base`: DinoV3-B (ViT-B/14, 86M 参数) - **推荐**
- `large`: DinoV3-L (ViT-L/14, 300M 参数)
- `giant2`: DinoV3-g (ViT-g/14, 1.1B 参数)
- `all`: 下载所有模型

权重文件将保存到 `weights/` 目录：
- `dinov3_vits14_pretrain.pth` (Small)
- `dinov3_vitb14_pretrain.pth` (Base)
- `dinov3_vitl14_pretrain.pth` (Large)
- `dinov3_vitg14_pretrain.pth` (Giant2)

### 3. 安装 DinoV3 依赖

进入 DinoV3 vendor 目录并安装依赖：

```bash
cd vfmkd/dinov3
pip install -e .
```

或者根据 DinoV3 的 README 安装所需依赖。

### 4. 验证安装

检查 DinoV3 vendor 目录和权重文件：

```bash
# 检查 vendor 目录
ls -la vfmkd/dinov3/

# 检查权重文件
ls -lh weights/dinov3_*.pth
```

## 📖 使用方法

### 基本使用

```python
import yaml
from vfmkd.teachers import DinoV3Teacher

# 加载配置
with open('configs/teachers/dinov3.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建 DinoV3 教师模型
teacher = DinoV3Teacher(config)

# 提取特征
import numpy as np
from PIL import Image

# 加载图像
image = np.array(Image.open('path/to/image.jpg'))

# 提取特征
features = teacher.extract_features(image)

# features 包含:
# - 'cls_token': CLS token 特征 (1, 768)
# - 'patch_tokens': Patch tokens 特征 (1, N, 768)
```

### 配置说明

编辑 `configs/teachers/dinov3.yaml` 来配置 DinoV3 教师模型：

```yaml
# 模型类型
model_type: "vit_base"  # vit_small, vit_base, vit_large, vit_giant2

# 权重路径
checkpoint_path: "weights/dinov3_vitb14_pretrain.pth"

# 特征提取配置
extract_cls_token: true      # 是否提取 CLS token
extract_patch_tokens: true   # 是否提取 patch tokens

# 设备配置
device: "cuda"

# 特征存储配置
feature_output_dir: "teacher_features/dino/"
```

## 🏗️ 架构说明

### 目录结构

```
vfmkd/
├── dinov3/                    # DinoV3 vendor 目录
│   └── ...                    # DinoV3 官方代码
├── teachers/
│   ├── base_teacher.py
│   ├── sam2_teacher.py
│   └── dinov3_teacher.py      # DinoV3 teacher 实现
├── configs/
│   └── teachers/
│       └── dinov3.yaml        # DinoV3 配置
└── weights/
    ├── dinov3_vits14_pretrain.pth
    ├── dinov3_vitb14_pretrain.pth
    ├── dinov3_vitl14_pretrain.pth
    └── dinov3_vitg14_pretrain.pth
```

### DinoV3 Teacher API

`DinoV3Teacher` 继承自 `BaseTeacher`，提供以下方法：

- `extract_features(images, image_ids, save_features)`: 提取特征
- `forward(x)`: 前向传播
- `get_feature_types()`: 返回特征类型列表
- `get_feature_dims()`: 返回特征维度字典
- `get_feature_strides()`: 返回特征下采样倍数
- `get_model_info()`: 返回模型详细信息

## 🔍 特征说明

DinoV3 提供两种类型的特征：

1. **CLS Token**: 全局图像表示
   - 形状: `(B, D)`，其中 D 取决于模型大小
   - 维度: Small=384, Base=768, Large=1024, Giant2=1536

2. **Patch Tokens**: 局部图像补丁表示
   - 形状: `(B, N, D)`，其中 N 是补丁数量
   - 对于 518x518 输入，N ≈ (518/14)² ≈ 1369 个补丁

## ⚠️ 注意事项

1. **首次使用**: 如果 DinoV3 vendor 目录不存在，需要先运行 `vendor_dinov3.py`
2. **权重文件**: 确保权重文件路径正确，否则会尝试从 HuggingFace 或官方仓库下载
3. **内存要求**: 
   - Base 模型: 约 330MB 权重
   - Large 模型: 约 1.1GB 权重
   - Giant2 模型: 约 4.1GB 权重
4. **依赖项**: DinoV3 可能需要特定的依赖项，请参考其官方文档

## 🐛 故障排除

### 问题: 无法导入 DinoV3 模块

**解决方案**:
1. 检查 vendor 目录是否存在: `ls vfmkd/dinov3/`
2. 重新运行 vendor 脚本: `python tools/vendor_dinov3.py --force`
3. 安装 DinoV3 依赖: `cd vfmkd/dinov3 && pip install -e .`

### 问题: 权重文件未找到

**解决方案**:
1. 检查权重文件路径: `ls weights/dinov3_*.pth`
2. 重新下载权重: `python tools/download_dinov3_weights.py --model base`
3. 检查配置文件中的 `checkpoint_path` 是否正确

### 问题: CUDA 内存不足

**解决方案**:
- 使用较小的模型 (Small 或 Base)
- 减小批次大小
- 使用 CPU: 在配置中设置 `device: "cpu"`

## 📚 参考资源

- [DinoV3 GitHub 仓库](https://github.com/facebookresearch/dinov3)
- [DinoV3 官方文档](https://dinov3.metademolab.com/)
- [Hugging Face DinoV3 模型库](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)

## 📝 更新日志

- 2024-XX-XX: 初始集成 DinoV3 支持


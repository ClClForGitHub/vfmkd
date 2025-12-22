# RT-DETRv2 使用说明

## 概述

本文档说明如何使用蒸馏得到的backbone权重来训练RT-DETRv2模型。

## 目录结构

```
RT-DETR-main/
├── RT-DETR-main/
│   └── rtdetrv2_pytorch/          # RT-DETRv2 PyTorch实现
│       ├── tools/
│       │   └── train.py            # 训练脚本
│       ├── configs/               # 配置文件目录
│       └── src/                   # 源代码
├── convert_backbone_weights.py    # 权重转换脚本
└── RT-DETRv2_使用说明.md          # 本文档
```

## 权重转换

### 1. 转换backbone权重

蒸馏得到的backbone权重文件中的键名格式为：`stem.stem1.conv.weight`，但RT-DETRv2模型期望的键名格式为：`backbone.stem.stem1.conv.weight`。

使用提供的转换脚本进行转换：

```bash
cd /home/team/zouzhiyuan/vfmkd/tools/core/RT-DETR-main

python convert_backbone_weights.py \
    --input /home/team/zouzhiyuan/vfmkd/outputs/distill_single_test_FGD/20251125_204817_rtdetrv2_edge_boost_fgd_gpu4_edge_boost_rtdetrv2/models/best_backbone_mmdet.pth \
    --output ./rtdetrv2_backbone_weights.pth
```

### 2. 权重文件格式说明

转换后的权重文件格式：
```python
{
    'model': {
        'backbone.stem.stem1.conv.weight': tensor(...),
        'backbone.stem.stem1.bn.weight': tensor(...),
        # ... 其他backbone权重
    }
}
```

RT-DETRv2的`load_tuning_state`函数会：
1. 检查checkpoint中是否有`ema`键
2. 如果没有`ema`，则使用`model`键
3. 匹配当前模型的state_dict键名，只加载匹配的权重
4. 忽略不匹配或缺失的键（strict=False）

## 训练RT-DETRv2

### 1. 配置文件

RT-DETRv2使用YAML配置文件。对于HGNetv2-L backbone，使用以下配置：

```yaml
# configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml
RTDETR:
  backbone: HGNetv2

HGNetv2:
  name: 'L'
  return_idx: [1, 2, 3]
  freeze_at: 0
  freeze_norm: True
  pretrained: False  # 设置为False，因为我们将通过tuning加载权重
```

### 2. 训练命令

使用`-t`参数指定tuning权重文件：

```bash
cd /home/team/zouzhiyuan/vfmkd/tools/core/RT-DETR-main/RT-DETR-main/rtdetrv2_pytorch

# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml \
    -t /path/to/rtdetrv2_backbone_weights.pth \
    --use-amp \
    --seed=0

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=9909 \
    --nproc_per_node=4 \
    tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml \
    -t /path/to/rtdetrv2_backbone_weights.pth \
    --use-amp \
    --seed=0 \
    &> log.txt 2>&1 &
```

### 3. 训练参数说明

- `-c, --config`: 配置文件路径
- `-t, --tuning`: tuning权重文件路径（用于加载backbone权重）
- `-r, --resume`: 恢复训练（加载完整checkpoint，包括optimizer、scheduler等）
- `--use-amp`: 启用自动混合精度训练
- `--seed`: 随机种子
- `--test-only`: 仅测试模式
- `--output-dir`: 输出目录（可在配置文件中设置）

**注意**：`-t`和`-r`不能同时使用。

## 代码逻辑说明

### 1. 模型构建流程

1. **YAMLConfig加载配置** (`src/core/yaml_config.py`)
   - 解析YAML配置文件
   - 通过`create`函数构建模型组件

2. **模型实例化** (`src/zoo/rtdetr/rtdetr.py`)
   ```python
   class RTDETR(nn.Module):
       def __init__(self, backbone, encoder, decoder):
           self.backbone = backbone  # backbone作为子模块
           self.encoder = encoder
           self.decoder = decoder
   ```

3. **权重加载** (`src/solver/_solver.py`)
   ```python
   def load_tuning_state(self, path: str):
       # 加载checkpoint
       state = torch.load(path, map_location='cpu')
       
       # 检查是否有ema，否则使用model
       if 'ema' in state:
           params = state['ema']['module']
       else:
           params = state['model']
       
       # 匹配键名并加载
       stat, infos = self._matched_state(module.state_dict(), params)
       module.load_state_dict(stat, strict=False)
   ```

### 2. 键名匹配机制

`_matched_state`函数会：
- 遍历当前模型的state_dict
- 检查checkpoint中是否有对应的键
- 检查形状是否匹配
- 只加载匹配的权重，忽略不匹配或缺失的键

这意味着：
- ✅ 只加载backbone权重（键名匹配`backbone.*`）
- ✅ encoder和decoder保持随机初始化
- ✅ 忽略形状不匹配的键（如BN的running_mean等，如果形状不同）

## 常见问题

### Q1: 如何确认权重加载成功？

在训练日志中会看到类似输出：
```
tuning checkpoint from /path/to/rtdetrv2_backbone_weights.pth
Load model.state_dict, {'missed': [...], 'unmatched': [...]}
```

- `missed`: 当前模型中有但checkpoint中没有的键（通常是encoder/decoder）
- `unmatched`: 键名匹配但形状不匹配的键

### Q2: 如何只加载部分backbone权重？

如果需要选择性加载，可以修改转换脚本，在转换时过滤键名：

```python
# 只加载特定层的权重
filtered_state_dict = {
    k: v for k, v in state_dict.items() 
    if 'stem' in k or 'stages.0' in k
}
```

### Q3: 如何冻结backbone参数？

在配置文件中设置：
```yaml
HGNetv2:
  freeze_at: 0      # 冻结前0层（不冻结）
  freeze_norm: True # 冻结BN层
```

或在optimizer配置中使用正则表达式：
```yaml
optimizer:
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0  # 设置学习率为0来冻结
```

### Q4: 权重文件格式不匹配怎么办？

如果权重文件格式与预期不同，可以修改转换脚本：

```python
# 如果权重文件直接是state_dict
if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
    # 检查是否是直接的state_dict
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
```

## 参考资源

- RT-DETRv2官方仓库: `RT-DETR-main/rtdetrv2_pytorch/`
- 配置文件示例: `configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml`
- 训练脚本: `tools/train.py`
- 权重加载逻辑: `src/solver/_solver.py::load_tuning_state`

## 完整示例

```bash
# 1. 转换权重
cd /home/team/zouzhiyuan/vfmkd/tools/core/RT-DETR-main
python convert_backbone_weights.py \
    --input /home/team/zouzhiyuan/vfmkd/outputs/distill_single_test_FGD/20251125_204817_rtdetrv2_edge_boost_fgd_gpu4_edge_boost_rtdetrv2/models/best_backbone_mmdet.pth \
    --output ./rtdetrv2_backbone_weights.pth

# 2. 训练模型
cd RT-DETR-main/rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=9909 \
    --nproc_per_node=4 \
    tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml \
    -t ../../rtdetrv2_backbone_weights.pth \
    --use-amp \
    --seed=0 \
    --output-dir ./output/rtdetrv2_hgnetv2_l_custom_backbone
```


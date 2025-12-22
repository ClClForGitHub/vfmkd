# SAM2模型切换指南

## 支持的模型
- sam2.1_hiera_t (Tiny)
- sam2.1_hiera_s (Small)
- sam2.1_hiera_b+ (Base+) - 默认
- sam2.1_hiera_l (Large)

## 切换方法

### 方法1: 修改配置文件
编辑 `configs/teachers/sam2.yaml`:
```yaml
model_type: "sam2.1_hiera_l"
checkpoint_path: "weights/sam2.1_hiera_large.pt"
```

编辑 `vfmkd/teachers/sam2_teacher.py` 第96行:
```python
config_name = f"sam2.1/sam2.1_hiera_l.yaml"
```

### 方法2: 命令行参数（推荐）
```bash
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco128 \
    --teacher sam2 \
    --model-type sam2.1_hiera_l \
    --checkpoint weights/sam2.1_hiera_large.pt
```

## 模型对比
| 模型 | 参数量 | 文件大小 | 性能 |
|------|--------|----------|------|
| Tiny | ~38M | ~150MB | 快速 |
| Small | ~46M | ~180MB | 平衡 |
| Base+ | ~81M | ~324MB | 推荐 |
| Large | ~224M | ~898MB | 最佳 |

## 使用示例

### 基本用法
```bash
# 使用SAM2 base+提取COCO128特征
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco128 \
    --teacher sam2

# 切换到large模型
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco128 \
    --teacher sam2 \
    --model-type sam2.1_hiera_l \
    --checkpoint weights/sam2.1_hiera_large.pt

# 处理完整COCO数据集
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco \
    --teacher sam2
```

### 高级选项
```bash
# 启用可视化（每50张图像）
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco128 \
    --teacher sam2 \
    --visualize true \
    --vis-interval 50

# 断点续传
python tools/prepare_teacher_features.py \
    --config configs/experiments/example.yaml \
    --dataset coco128 \
    --teacher sam2 \
    --skip-existing
```

## 注意事项

1. **权重文件**: 确保对应的.pt权重文件存在于weights/目录
2. **配置文件**: 确保SAM2 vendor中有对应的yaml配置文件
3. **显存要求**: Large模型需要更多显存，建议至少8GB
4. **性能权衡**: 根据任务需求选择合适的模型大小


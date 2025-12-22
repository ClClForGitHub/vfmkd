# 图像预处理快速参考

## 🎯 一句话总结
**SAM2用官方预处理（含ImageNet标准化），YOLO用简化预处理（仅/255），Adapter负责对齐！**

---

## 📋 快速对比表

| 模块 | 预处理流程 | 代码位置 | 用途 |
|------|-----------|---------|------|
| **SAM2 Teacher** | `Resize(1024) + ToTensor + ImageNet Norm` | `sam2_teacher.py:327-337` | 离线提取特征存NPZ |
| **YOLO Student** | `Resize(1024) + ToTensor` | `train_adapter_align.py:56-58` | 在线训练输入 |
| **Adapter** | （无预处理，接收特征） | `vfmkd/models/heads/` | 特征空间对齐 |

---

## 🔧 代码示例

### SAM2 Teacher (特征提取)
```python
# vfmkd/teachers/sam2_teacher.py
from sam2.utils.transforms import SAM2Transforms

# 初始化transform
self._sam2_transforms = SAM2Transforms(
    resolution=1024,  # Resize到1024x1024
    mask_threshold=0.0,
    max_hole_area=0.0,
    max_sprinkle_area=0.0
)

# 应用transform（自动：Resize + ToTensor + ImageNet Normalize）
image_tensor = self._sam2_transforms(image).unsqueeze(0)
```

### YOLO Student (训练输入)
```python
# tools/train_adapter_align.py
from PIL import Image
import torchvision.transforms.functional as TF

# Resize到1024x1024
img = Image.open(img_path).convert('RGB').resize((1024, 1024))

# ToTensor（仅/255.0，无ImageNet标准化）
x = TF.to_tensor(img)  # 输出: [0, 1]
```

---

## ❓ 5秒决策树

```
需要预处理图像？
│
├─ 是否使用SAM2预训练权重？
│  ├─ 是 → 用 SAM2Transforms（含ImageNet Norm）
│  └─ 否 → 继续判断
│
└─ 是否YOLO架构？
   ├─ 是 → 用 Resize + ToTensor（仅/255）
   └─ 否 → 根据预训练方式决定
```

---

## ✅ 当前实现状态

- ✅ SAM2 Teacher: 已使用`SAM2Transforms`
- ✅ YOLO Student: 已使用`Resize + ToTensor`
- ✅ 训练脚本: 已添加清晰注释
- ✅ 文档: 已创建`PREPROCESSING_STRATEGY.md`
- ✅ 记忆: 已存入长期记忆（ID: 10596519）

---

## 🚨 注意事项

1. **不要混用**：同一模块内保持预处理一致
2. **NPZ特征**：已用SAM2预处理提取，训练时不要重新提取
3. **推理时**：仅用YOLO预处理，不需要SAM2 Image Encoder
4. **修改前**：先确认是否影响已训练的Adapter权重

---

## 📚 详细文档
➡️ 完整说明见 [`PREPROCESSING_STRATEGY.md`](./PREPROCESSING_STRATEGY.md)


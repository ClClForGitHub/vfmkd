# 图像预处理策略文档

## 📋 概述

本项目采用**双轨预处理策略**：
- **SAM2 Teacher（教师模型）**：使用SAM2官方预处理（含ImageNet标准化）
- **YOLO Student（学生模型）**：使用简化预处理（仅Resize+/255）
- **Adapter（适配器）**：负责跨特征空间的对齐

---

## 🎯 设计原则

### 1. **SAM2 Teacher预处理**
**目的**：最大化利用SAM2预训练权重

**流程**（`SAM2Transforms`）：
```python
# 1. Resize到1024x1024（双线性插值）
image = torchvision.transforms.Resize((1024, 1024))(image)

# 2. ToTensor（转为CHW格式并/255.0归一化）
image = torchvision.transforms.ToTensor()(image)  # [0,1]

# 3. ImageNet标准化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = torchvision.transforms.Normalize(mean, std)(image)
```

**应用场景**：
- ✅ SAM2特征提取（`sam2_teacher.py`）
- ✅ SAM2掩码预测（`predictor.set_image`）
- ✅ 所有使用SAM2预训练权重的场景

---

### 2. **YOLO Student预处理**
**目的**：保持YOLO架构的标准输入格式

**流程**：
```python
# 1. Resize到1024x1024
from PIL import Image
image = Image.open(path).convert('RGB').resize((1024, 1024))

# 2. ToTensor（仅/255.0归一化，无ImageNet标准化）
import torchvision.transforms.functional as TF
image_tensor = TF.to_tensor(image)  # [0,1]
```

**应用场景**：
- ✅ YOLO backbone训练（`train_adapter_align.py`）
- ✅ YOLOv8特征提取
- ✅ 所有YOLO架构的学生模型

---

### 3. **Adapter的作用**
**目的**：跨特征空间对齐

```
输入图像 (原始图片)
    │
    ├──> SAM2预处理 ──> SAM2 Encoder ──> Teacher Features (含ImageNet统计)
    │                                            │
    │                                            ↓
    └──> YOLO预处理 ──> YOLO Backbone ──> Student Features (仅/255)
                                                 │
                                                 ↓
                                            Adapter对齐
                                                 │
                                                 ↓
                                        对齐后的Student Features
                                                 │
                                                 ↓
                                        特征蒸馏Loss（与Teacher对比）
```

**关键点**：
- ✅ Adapter学习的是**特征空间映射**，而非图像预处理差异
- ✅ 训练时，Student和Teacher用不同预处理是合理的
- ✅ Adapter通过可学习参数（1x1卷积、LayerNorm等）实现对齐

---

## 📊 数据流图

### 训练阶段
```
图像文件 (原始JPG/PNG)
    │
    ├──> [离线] SAM2官方预处理 ──> SAM2 ──> NPZ (Teacher Features)
    │
    └──> [在线] YOLO预处理 ──> YOLO Backbone ──> Adapter ──> 对齐特征
                                                              │
                                                              ↓
                                                   MSE/Cosine Loss
                                                   与NPZ对比训练
```

### 推理阶段
```
图像文件
    │
    └──> YOLO预处理 ──> YOLO Backbone ──> Adapter ──> 特征
                                                        │
                                                        ↓
                                              SAM2 Prompt Encoder
                                                        │
                                                        ↓
                                              SAM2 Mask Decoder
                                                        │
                                                        ↓
                                                  分割掩码输出
```

**注意**：推理时不需要SAM2 Image Encoder，只用YOLO+Adapter替代！

---

## 🔧 实现文件

### 1. SAM2 Teacher
- **文件**: `vfmkd/teachers/sam2_teacher.py`
- **关键代码**:
  ```python
  # 第327-334行
  from sam2.utils.transforms import SAM2Transforms
  self._sam2_transforms = SAM2Transforms(
      resolution=1024,
      mask_threshold=0.0,
      max_hole_area=0.0,
      max_sprinkle_area=0.0
  )
  image_tensor = self._sam2_transforms(image).unsqueeze(0)
  ```

### 2. YOLO Student训练
- **文件**: `tools/train_adapter_align.py`
- **关键代码**:
  ```python
  # 第56-58行
  img = Image.open(img_path).convert('RGB').resize((1024, 1024))
  x = TF.to_tensor(img)  # 自动 /255.0
  ```

### 3. Warmup训练
- **文件**: `tools/warmup_training_v1.py`
- **关键代码**:
  ```python
  # 第98-102行
  image_resized = cv2.resize(image_rgb, (1024, 1024))
  image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
  ```

---

## ❓ 常见问题

### Q1: 为什么Teacher和Student用不同的预处理？
**A**: 
- SAM2是预训练模型，必须用ImageNet标准化才能发挥最佳性能
- YOLO是从头训练或微调，使用简单的/255归一化即可
- Adapter的作用就是弥合这两种特征空间的差异

### Q2: 这样训练会有问题吗？
**A**: 不会！这是标准的知识蒸馏实践：
- Teacher用最优配置提取最佳特征
- Student用自己的输入格式
- 通过可学习的Adapter建立映射关系

### Q3: 推理时用哪种预处理？
**A**: **仅用YOLO预处理**（Resize+/255）
- 输入图像 → YOLO预处理 → YOLO Backbone → Adapter → 特征
- 特征 → SAM2 Prompt/Mask Decoder → 分割掩码
- 不需要SAM2 Image Encoder，所以不需要SAM2预处理

### Q4: NPZ中的Teacher特征是哪种预处理提取的？
**A**: **SAM2官方预处理**（含ImageNet标准化）
- 离线提取时使用`SAM2Transforms`
- 保存在NPZ中，训练时直接加载
- Student通过Adapter学习对齐到这个特征空间

### Q5: 如果我要用其他backbone（如RepViT）怎么办？
**A**: 
- **如果是轻量级模型（RepViT、MobileNet等）**：使用YOLO预处理（Resize+/255）
- **如果是预训练大模型（ResNet、EfficientNet等）**：使用SAM2预处理（含ImageNet标准化）
- **核心原则**：与backbone的预训练方式保持一致

---

## ✅ 验证清单

在实现或修改预处理时，请确认：

- [ ] SAM2 Teacher使用`SAM2Transforms`（含ImageNet标准化）
- [ ] YOLO Student训练使用`Resize + ToTensor`（仅/255）
- [ ] NPZ特征是用SAM2官方预处理提取的
- [ ] Adapter训练时同时处理两种预处理的特征
- [ ] 推理时仅用YOLO预处理+Adapter+SAM2 Decoder
- [ ] 所有注释清楚说明当前使用的预处理方式

---

## 📚 相关记忆

根据用户长期记忆（Memory ID: 10583979）：
- 使用错误的`predictor.get_image_embedding()`会导致特征permute/view变换
- 正确方式：`backbone_out = sam2_model.image_encoder(image_tensor); vision_features = backbone_out['vision_features']`
- 使用正确特征后，RepViT+Adapter的IOU从0.65提升到0.9372

---

## 🔄 更新日志

**2025-10-31**:
- 初始版本，明确双轨预处理策略
- SAM2使用官方预处理（含ImageNet标准化）
- YOLO使用简化预处理（仅Resize+/255）
- Adapter负责特征空间对齐

---

**维护者注意**：
任何修改预处理流程的PR都必须更新此文档！


# CLIP模型使用说明

## 📋 模型信息

### 模型类型
- **模型架构**: `ViT-B-32` (Vision Transformer Base, 32x32 patch size)
- **预训练数据集**: `laion2b_s34b_b79k` (LAION-2B数据集)
- **权重来源**: OpenCLIP (https://github.com/mlfoundations/open_clip)
- **权重格式**: 
  - `open_clip_model.safetensors` (优先)
  - `open_clip_pytorch_model.bin` (备选)

### 权重路径
```
/home/team/zouzhiyuan/vfmkd/weights/clip/ViT-B-32-laion2B-s34B-b79K/
├── open_clip_model.safetensors
└── open_clip_pytorch_model.bin
```

## 🔧 使用方法

### 1. 模型加载

```python
import open_clip
import torch
from safetensors.torch import load_file as load_safetensor
from pathlib import Path

# 设备配置（默认使用cuda:4）
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# 权重路径
weight_dir = Path("weights/clip/ViT-B-32-laion2B-s34B-b79K")
weight_path = weight_dir / "open_clip_model.safetensors"  # 或 open_clip_pytorch_model.bin

# 创建模型（不使用pretrained，避免网络请求）
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained=None,  # 关键：不使用pretrained，避免网络请求
    device=device
)

# 加载本地权重
if weight_path.suffix == '.safetensors':
    state_dict = load_safetensor(str(weight_path))
else:
    state_dict = torch.load(str(weight_path), map_location=device)

model.load_state_dict(state_dict, strict=False)
model.eval()
```

### 2. 文本编码

```python
# 获取tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 定义提示词
positive_prompts = [
    'a person', 'a man', 'a woman', 'a child', 'a group of people',
    'a bicycle', 'a car', 'a motorcycle', 'a bus', 'a truck',
    'a building', 'a room', 'an object'
    # ... 完整的COCO 80类别
]

negative_prompts = [
    'sky', 'ground', 'road', 'wall', 'grass', 'tree', 'background',
    'a sleeve', 'a pant leg', 'clothing', 'a texture', 'a part of', 'a fragment'
    # ... 更多背景/片段提示词
]

# 编码文本特征
with torch.no_grad():
    tokens_pos = tokenizer(positive_prompts).to(device)
    tokens_neg = tokenizer(negative_prompts).to(device)
    
    text_features_pos = model.encode_text(tokens_pos)
    text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)  # L2归一化
    
    text_features_neg = model.encode_text(tokens_neg)
    text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)  # L2归一化
```

### 3. 图像预处理

```python
from PIL import Image
import cv2
import numpy as np

# 方法1: 原始方法（掩码外置灰128，掩码内×1.05）
def preprocess_original(cropped_pil: Image.Image, mask_resized: np.ndarray) -> Image.Image:
    cropped_np = np.array(cropped_pil).astype(np.float32)
    background_value = 128.0
    mask_zero = mask_resized == 0
    cropped_np[mask_zero] = background_value
    
    mask_positive = mask_resized > 0
    if mask_positive.any():
        cropped_np[mask_positive] *= 1.05
    cropped_np = np.clip(cropped_np, 0.0, 255.0)
    return Image.fromarray(cropped_np.astype(np.uint8))

# 方法2: 原始裁剪（不处理）
def preprocess_raw_crop(cropped_pil: Image.Image, mask_resized: np.ndarray) -> Image.Image:
    return cropped_pil

# 方法3: 软遮罩（alpha blending）
def preprocess_soft_mask(cropped_pil: Image.Image, mask_resized: np.ndarray, alpha: float = 0.4) -> Image.Image:
    cropped_np = np.array(cropped_pil).astype(np.float32)
    background_value = 128.0
    background = np.full_like(cropped_np, background_value)
    
    mask_positive = mask_resized > 0.5
    mask_negative = ~mask_positive
    
    # 掩码外区域：原图 * (1-alpha) + 灰色 * alpha
    cropped_np[mask_negative] = (
        cropped_np[mask_negative] * (1 - alpha) + 
        background[mask_negative] * alpha
    )
    
    # 掩码内区域：稍微增强
    if mask_positive.any():
        cropped_np[mask_positive] *= 1.05
    
    cropped_np = np.clip(cropped_np, 0.0, 255.0)
    return Image.fromarray(cropped_np.astype(np.uint8))

# 使用preprocess函数转换为模型输入
processed_pil = preprocess_original(cropped_pil, mask_resized)
image_tensor = preprocess(processed_pil).unsqueeze(0).to(device)  # [1, 3, H, W]
```

### 4. 图像编码与相似度计算

```python
# 批量处理图像
image_tensors = [preprocess(img).to(device) for img in processed_images]
image_batch_tensor = torch.stack(image_tensors, dim=0)  # [N, 3, H, W]

# 编码图像特征
with torch.no_grad():
    image_features = model.encode_image(image_batch_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2归一化
    
    # 计算相似度（余弦相似度 = 归一化特征的点积）
    sim_pos = image_features @ text_features_pos.T  # [N, num_positive_prompts]
    sim_neg = image_features @ text_features_neg.T  # [N, num_negative_prompts]
    
    # 找到最高分
    pos_scores, pos_idxs = sim_pos.max(dim=1)  # [N], [N]
    neg_scores, neg_idxs = sim_neg.max(dim=1)  # [N], [N]
```

## 📊 返回参数说明

### 1. `model.encode_image(image_tensor)` 返回值

**输入**:
- `image_tensor`: `torch.Tensor`, shape `[N, 3, H, W]` 或 `[3, H, W]`
  - 经过`preprocess`函数处理后的图像张量
  - 值域: 已归一化（ImageNet标准化）

**输出**:
- `image_features`: `torch.Tensor`, shape `[N, 512]` 或 `[512]`
  - 图像特征向量（未归一化）
  - 维度: 512 (ViT-B-32的embedding维度)

**示例**:
```python
image_features = model.encode_image(image_tensor)
print(f"Shape: {image_features.shape}")  # [N, 512]
print(f"Mean: {image_features.mean():.4f}, Std: {image_features.std():.4f}")
```

### 2. `model.encode_text(text_tokens)` 返回值

**输入**:
- `text_tokens`: `torch.Tensor`, shape `[N, seq_len]`
  - 经过tokenizer编码的文本token序列
  - 通常`seq_len=77`（CLIP的最大序列长度）

**输出**:
- `text_features`: `torch.Tensor`, shape `[N, 512]`
  - 文本特征向量（未归一化）
  - 维度: 512 (与图像特征维度相同)

**示例**:
```python
tokens = tokenizer(["a person", "a car"])
text_features = model.encode_text(tokens)
print(f"Shape: {text_features.shape}")  # [2, 512]
```

### 3. 原生CLIP分类（推荐方式）

**使用 logit_scale + softmax 得到概率**:
```python
# 归一化特征
image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

# 计算 logits（原生 CLIP 方式）
logit_scale = model.logit_scale.exp()  # 温度参数（通常几十到一百多）
logits = logit_scale * image_features_norm @ text_features_norm.T  # [N_images, N_texts]

# softmax 得到概率
probs = logits.softmax(dim=-1)  # [N_images, N_texts]，每行和为1.0
```

**概率范围**:
- 理论上: `[0.0, 1.0]` (softmax后的概率)
- 所有类别的概率和为 1.0
- 高置信度: 最高概率 `> 0.5` 通常表示模型很确定

**为什么用概率而不是余弦相似度？**
- 概率有明确的语义：表示"这个图像属于某个类别的可能性"
- 概率值在 [0, 1] 范围内，更容易理解和设置阈值
- 符合CLIP原生的zero-shot分类流程

### 4. 组件打分结果 (`comp` 字典)

在`test_bbox_strategies.py`中，每个组件会添加以下CLIP相关字段:

```python
comp = {
    # ... 其他字段 ...
    
    # CLIP语义打分（使用原生概率）
    's_pos': float,              # 正类最高概率 (0.0-1.0)
    's_neg': float,              # 负类最高概率 (0.0-1.0)
    's_pos_text': str,          # 匹配的正类提示词 (如 "a person")
    's_neg_text': str,           # 匹配的负类提示词 (如 "ground")
    'semantic_multiplier': float, # 语义乘数 (2.0 / 1.0 / 0.5 / 0.1)
}
```

**`semantic_multiplier` 规则**（基于概率总和）:
- `2.0`: `p_pos_sum > 0.7` 且 `p_neg_sum < 0.2` (很确定是前景目标)
- `1.0`: `p_pos_sum > 0.4` (中等置信度正类)
- `0.5`: 其他情况（模糊/不确定）
- `0.1`: `p_neg_sum > 0.5` (明显是背景/碎片)

**概率统计量**:
- `p_pos_sum`: 所有正类提示词的概率总和
- `p_neg_sum`: 所有负类提示词的概率总和
- `p_pos_max`: 正类中最高概率
- `p_neg_max`: 负类中最高概率

## 📝 完整使用示例（原生CLIP方式）

```python
import torch
import open_clip
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file as load_safetensor

# 1. 加载模型
device = torch.device('cuda:4')
weight_path = Path("weights/clip/ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors")

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, device=device)
state_dict = load_safetensor(str(weight_path))
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. 准备文本特征
tokenizer = open_clip.get_tokenizer('ViT-B-32')
positive_prompts = ['a person', 'a car', 'a building']
negative_prompts = ['sky', 'ground', 'background']
all_prompts = positive_prompts + negative_prompts
num_pos = len(positive_prompts)

tokens = tokenizer(all_prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 3. 处理图像
image_pil = Image.open("image.jpg")
image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

# 4. 编码图像并使用原生CLIP方式计算概率
with torch.no_grad():
    # 编码图像
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 计算 logits（原生 CLIP 方式）
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T  # [1, num_prompts]
    
    # softmax 得到概率
    probs = logits.softmax(dim=-1)  # [1, num_prompts]
    probs = probs[0]  # [num_prompts]
    
    # 分离正类和负类概率
    pos_probs = probs[:num_pos]
    neg_probs = probs[num_pos:]
    
    # 计算统计量
    p_pos_sum = pos_probs.sum().item()
    p_neg_sum = neg_probs.sum().item()
    p_pos_max, pos_max_idx = pos_probs.max(dim=0)
    p_neg_max, neg_max_idx = neg_probs.max(dim=0)
    top_prob, top_idx = probs.max(dim=0)
    
    print(f"最高概率: {top_prob.item():.4f}")
    print(f"匹配提示词: {all_prompts[top_idx.item()]}")
    print(f"正类概率总和: {p_pos_sum:.4f}")
    print(f"负类概率总和: {p_neg_sum:.4f}")
    print(f"正类最高概率: {p_pos_max.item():.4f} ({positive_prompts[pos_max_idx.item()]})")
    print(f"负类最高概率: {p_neg_max.item():.4f} ({negative_prompts[neg_max_idx.item()]})")
    
    # 根据概率设计语义乘数
    if p_neg_sum > 0.5:
        semantic_multiplier = 0.1
    elif p_pos_sum > 0.7 and p_neg_sum < 0.2:
        semantic_multiplier = 2.0
    elif p_pos_sum > 0.4:
        semantic_multiplier = 1.0
    else:
        semantic_multiplier = 0.5
    
    print(f"语义乘数: {semantic_multiplier}")
```

## ⚠️ 注意事项

1. **权重加载**: 必须使用`pretrained=None`创建模型，然后手动加载本地权重，避免网络请求
2. **特征归一化**: 计算logits前必须对图像和文本特征进行L2归一化
3. **使用logit_scale**: **必须**使用`model.logit_scale.exp()`作为温度参数，这是CLIP原生方式的关键
4. **使用softmax**: 必须对logits做softmax得到概率，而不是直接使用余弦相似度
5. **批处理**: `encode_image`和`encode_text`都支持批处理，可以提高效率
6. **设备一致性**: 确保所有张量都在同一设备上（通常是GPU）
7. **推理模式**: 使用`model.eval()`和`torch.no_grad()`确保推理效率

## 🔄 从余弦相似度迁移到原生CLIP方式

**旧方式（不推荐）**:
```python
similarity = image_features @ text_features.T  # 余弦相似度 [-1, 1]
scores, indices = similarity.max(dim=1)
```

**新方式（推荐）**:
```python
logit_scale = model.logit_scale.exp()  # 温度参数
logits = logit_scale * image_features @ text_features.T  # logits
probs = logits.softmax(dim=-1)  # 概率 [0, 1]
scores, indices = probs.max(dim=1)
```

**优势**:
- 概率值有明确的语义（属于某个类别的可能性）
- 所有类别概率和为1.0，更容易理解和设置阈值
- 符合CLIP原生的zero-shot分类流程

## 🔗 相关文件

- 主使用文件: `tools/core/bbox/test_bbox_strategies.py`
- 测试脚本: `tools/core/bbox/test_clip_preprocessing.py`
- 权重目录: `weights/clip/ViT-B-32-laion2B-s34B-b79K/`


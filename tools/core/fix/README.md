# NPZ修复工具整理文档

## 📋 概述

本文档整理了项目中所有用于修复NPZ文件的工具脚本，包括特征修复和边缘修复两大类。这些工具用于批量更新或修复已生成的NPZ特征文件。

---

## 🔧 特征修复工具

### 1. `tools/core/fix/bulk_fix_npz_features.py`
**功能**: 批量替换NPZ中的P4/P5特征，并标记`feature_flag=1`

**实现方式**: 
- ✅ **内部实现** - 直接调用SAM2底层API
- 不依赖`SAM2Teacher`类，直接使用`build_sam2`和`SAM2Transforms`
- 使用`model.image_encoder()`提取特征，获取`backbone_fpn[2]`(P4)和`backbone_fpn[3]`(P5)

**特点**:
- 递归扫描指定目录下的所有NPZ文件（支持`*_features.npz`和`*_sam2_features.npz`）
- 自动去重（优先处理`*_features.npz`）
- 通过`feature_flag`标记跳过已处理文件
- 原子写入（临时文件+替换，避免写入中断损坏）

**关键代码**:
```python
# 加载SAM2模型（直接调用底层API）
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
model = build_sam2(config_file=cfg_name, ckpt_path=str(weights_path), device=str(device))
transforms = SAM2Transforms(resolution=model.image_size, ...)

# 提取特征
out = model.image_encoder(img_t)
p4 = out['backbone_fpn'][2]  # P4_S16
p5 = out['backbone_fpn'][3]  # P5_S32

# 更新NPZ
data['P4_S16'] = p4
data['P5_S32'] = p5
data['feature_flag'] = 1
```

**使用示例**:
```bash
python tools/core/fix/bulk_fix_npz_features.py \
    --root /home/team/zouzhiyuan/dataset/sa1b \
    --weights weights/sam2.1_hiera_base_plus.pt \
    --log logs/bulk_fix.log \
    --max-files 1000
```

---

### 2. `tools/core/fix/fix_npz_features_inplace.py`
**功能**: 原地修复NPZ特征（重算P4/P5），基于样本清单处理

**实现方式**:
- ✅ **内部实现** - 与`bulk_fix_npz_features.py`相同
- 直接调用SAM2底层API，不依赖`SAM2Teacher`
- 支持灵活的文件查找策略（同目录优先，然后搜索集中目录）

**特点**:
- 基于样本清单（每行一个jpg绝对路径）处理
- 支持NPZ文件分散存储（可通过`--npz-dirs`指定多个搜索目录）
- 同时更新`P4_S16`和`IMAGE_EMB_S16`（兼容键）
- 原子替换写入

**关键代码**:
```python
# 查找NPZ文件（多策略）
def find_npz_for_stem(stem: str, img_dir: Path, search_dirs: List[Path]):
    # 1. 优先同目录 _features.npz
    # 2. 其次同目录 _sam2_features.npz
    # 3. 在集中目录递归搜索

# 更新特征（兼容键）
data['P4_S16'] = p4
data['IMAGE_EMB_S16'] = p4  # 兼容键
data['P5_S32'] = p5
```

**使用示例**:
```bash
python tools/core/fix/fix_npz_features_inplace.py \
    --samples /path/to/sample_list.txt \
    --npz-dirs /path/to/npz1,/path/to/npz2 \
    --weights weights/sam2.1_hiera_base_plus.pt \
    --log logs/inplace_fix.log \
    --max-images 500
```

---

### 3. `tools/core/fix/fix_train_test_npz.py`
**功能**: 修复train_1200/test目录中extracted下的NPZ，图片位于父目录

**实现方式**:
- ✅ **内部实现** - 与上述两个脚本相同
- 直接调用SAM2底层API
- 专门针对train/test目录结构优化

**特点**:
- 支持多个extracted目录（逗号分隔）
- 自动从父目录查找图片文件（支持多种图片格式）
- 使用`shutil.move`替代`os.replace`（跨设备兼容）
- 遵循项目GPU策略（屏蔽0和3，优先1,2,4,5,6,7）

**关键代码**:
```python
# 解析图片路径（从extracted目录的父目录查找）
def resolve_image_path(extracted_dir: Path, stem: str):
    parent = extracted_dir.parent
    for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'):
        cand = parent / f'{stem}{ext}'
        if cand.exists():
            return cand

# 跨设备原子写入
def atomic_write_npz_shutil(target_path: Path, data_dict: dict):
    # 使用shutil.move替代os.replace（跨设备兼容）
    shutil.move(tmp_name, target_path)
```

**使用示例**:
```bash
python tools/core/fix/fix_train_test_npz.py \
    --extracted-dirs /path/train_1200/extracted,/path/test/extracted \
    --weights weights/sam2.1_hiera_base_plus.pt \
    --log logs/fix_train_test.log \
    --device cuda:4 \
    --max-files 1000
```

---

## 🎨 边缘修复工具

### 4. `tools/core/fix/update_edge_maps_from_npz.py`
**功能**: 批量更新NPZ文件中的边缘图（使用Method B）

**实现方式**:
- ✅ **完全内部实现** - 不依赖任何其他模块
- 直接从JSON文件提取边缘图（Method B）
- 使用OpenCV和pycocotools进行边缘提取

**特点**:
- 完全独立的实现，不调用`extract_features_v1.py`中的函数
- 使用Method B（每实例提取边缘后合并）
- 支持多尺度边缘图更新（256×256, 64×64, 32×32）
- 通过`edge_flag`和`edge_version`标记已处理文件
- 支持多种过滤和排序选项

**关键代码**:
```python
def extract_edges_method_b(json_path, kernel_size=3):
    """Method B：每个实例单独提取边缘后合并"""
    combined_edge_map = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        rle = ann['segmentation']
        mask = mask_utils.decode(rle)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        edge = (edge > 0).astype(np.uint8)
        combined_edge_map = np.bitwise_or(combined_edge_map, edge)
    
    # 生成多尺度边缘图
    for size in [256, 64, 32]:
        edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
        edge_maps[size] = (edge_small > 0).astype(np.uint8)
    
    return edge_maps

def update_single_npz_edge_maps(npz_path, json_dir, kernel_size=3):
    """更新单个NPZ文件的边缘图"""
    # 从NPZ文件名提取image_id
    image_id = npz_path.stem.replace('_features', '')
    json_path = Path(json_dir) / f"{image_id}.json"
    
    # 生成新的边缘图
    edge_maps = extract_edges_method_b(json_path, kernel_size)
    
    # 更新NPZ
    npz_data['edge_64x64'] = edge_maps[64]
    npz_data['edge_32x32'] = edge_maps[32]
    if 'edge_256x256' in npz_data:
        npz_data['edge_256x256'] = edge_maps[256]
    
    # 标记
    npz_data['edge_flag'] = np.array(1, dtype=np.uint8)
    npz_data['edge_version'] = np.array('B_v1')
```

**使用示例**:
```bash
python tools/core/fix/update_edge_maps_from_npz.py \
    --npz-dir /home/team/zouzhiyuan/dataset/sa1b/extracted \
    --json-dir /home/team/zouzhiyuan/dataset/sa1b \
    --kernel-size 3 \
    --max-files 1000 \
    --skip-if-processed \
    --set-edge-flag
```

---

## 📊 对比总结

| 工具 | 类型 | 实现方式 | 依赖 | Flag标记 | 特点 |
|------|------|----------|------|----------|------|
| `tools/core/fix/bulk_fix_npz_features.py` | 特征修复 | 内部实现（SAM2底层API） | `sam2.build_sam`, `SAM2Transforms` | `feature_flag=1` | 递归扫描，自动去重 |
| `tools/core/fix/fix_npz_features_inplace.py` | 特征修复 | 内部实现（SAM2底层API） | `sam2.build_sam`, `SAM2Transforms` | 无 | 基于样本清单，支持分散存储 |
| `tools/core/fix/fix_train_test_npz.py` | 特征修复 | 内部实现（SAM2底层API） | `sam2.build_sam`, `SAM2Transforms` | `feature_flag=1` | 针对train/test目录结构 |
| `tools/core/fix/update_edge_maps_from_npz.py` | 边缘修复 | 完全内部实现 | `cv2`, `pycocotools`, `numpy` | `edge_flag=1`, `edge_version='B_v1'` | 独立实现，不依赖其他模块 |

---

## 🔍 实现细节

### 特征修复脚本的共同点

1. **不调用SAM2Teacher**: 所有特征修复脚本都直接调用SAM2底层API，而不是使用`vfmkd.teachers.sam2_teacher.SAM2Teacher`类
2. **相同的特征提取流程**:
   ```python
   # 1. 加载模型
   model = build_sam2(config_file, ckpt_path, device)
   transforms = SAM2Transforms(resolution=model.image_size, ...)
   
   # 2. 预处理图像
   img_t = transforms(image_rgb).unsqueeze(0).to(device)
   
   # 3. 提取特征
   out = model.image_encoder(img_t)
   p4 = out['backbone_fpn'][2]  # P4_S16 (64×64)
   p5 = out['backbone_fpn'][3]  # P5_S32 (32×32)
   ```
3. **原子写入**: 所有脚本都使用临时文件+原子替换的方式，避免写入中断损坏文件

### 边缘修复脚本的特点

1. **完全独立**: 不调用`extract_features_v1.py`中的`extract_edges_and_weights_optimized`函数
2. **Method B实现**: 与`extract_features_v1.py`中的Method B完全一致，但独立实现
3. **标记机制**: 使用`edge_flag`和`edge_version`双重标记，便于版本管理和跳过已处理文件

---

## 💡 使用建议

1. **特征修复**: 优先使用`tools/core/fix/bulk_fix_npz_features.py`进行批量修复，如果NPZ文件分散存储，使用`tools/core/fix/fix_npz_features_inplace.py`
2. **边缘修复**: 使用`tools/core/fix/update_edge_maps_from_npz.py`，支持多种过滤选项，适合大规模批量更新
3. **Flag检查**: 所有工具都支持通过flag跳过已处理文件，提高效率
4. **日志记录**: 所有工具都支持日志输出，便于追踪处理进度和错误

---

## 🔗 相关文件

- `tools/core/extract_features_v1.py` - 原始特征提取脚本（这些修复工具的目标是修复该脚本生成的NPZ）
- `vfmkd/teachers/sam2_teacher.py` - SAM2Teacher类（特征修复脚本**不**使用该类，而是直接调用底层API）

---

**最后更新**: 2025-11-05


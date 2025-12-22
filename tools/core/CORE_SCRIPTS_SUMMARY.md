# Tools/Core 目录脚本功能汇总

**生成时间**: 2025-11-05  
**目录**: `tools/core/`

---

## 📁 目录结构

```
tools/core/
├── extract_features_v1.py              # 主特征提取脚本（核心）
├── extract_features_edge_comparison.py  # 边缘提取方法对比实验
├── train_distill_single_test.py        # 单图蒸馏训练测试
├── unified_model_test.py              # 统一模型测试评估
├── bbox/
│   └── sa1b_bbox_extractor.py         # SA-1B边界框提取器
├── fix/                                # NPZ修复工具集合
│   ├── bulk_fix_npz_features.py       # 批量修复特征
│   ├── fix_npz_features_inplace.py    # 原地修复特征
│   ├── fix_train_test_npz.py         # 修复train/test目录
│   └── update_edge_maps_from_npz.py  # 更新边缘图
└── prompt/                             # SAM2 Prompt相关工具
    ├── builder.py                      # Prompt构建器
    ├── prompts.py                      # Prompt工具函数
    ├── seg_head.py                     # 分割头测试
    └── test_*.py                       # 各种测试脚本
```

---

## 🔧 核心脚本详情

### 1. `extract_features_v1.py` ⭐ **核心工具**
**功能**: 主特征提取脚本（生产环境使用）

**主要功能**:
- 使用SAM2.1hiera教师模型提取16×下采样256通道特征
- 从SA-1B JSON生成多尺度边缘图（256×256, 64×64, 32×32）
- 生成前景/背景权重图（128×128, 64×64, 32×32）
- **Method B边缘提取**（每实例提取边缘后合并，更准确）
- **去重检查**（支持断点续传）
- 支持指定GPU设备（`--device`参数）
- 保存NPZ文件（特征 + 边缘图 + 权重图 + 元数据）

**特点**:
- 384行，功能完整
- 支持批量处理
- 详细的进度和计时统计
- 空间优化（不保存edge_original）

---

### 2. `extract_features_edge_comparison.py` 🔬 **实验脚本**
**功能**: 边缘提取方法对比实验

**主要功能**:
- 对比三种边缘提取方法：
  - **Method A**: Union mask then extract edge（基线方法）
  - **Method B**: Extract edge per instance then merge（改进方法1）
  - **Method C**: Instance mask map (different values) then morphology（改进方法2）
- 性能对比（速度、准确性）
- 可视化对比结果
- 生成对比报告

**特点**:
- 1067行，详细实验脚本
- 用于选择最佳边缘提取方法
- **Method B已被extract_features_v1.py采用**

**状态**: ⚠️ 实验脚本，可归档

---

### 3. `train_distill_single_test.py` 🧪 **测试脚本**
**功能**: 单图蒸馏训练测试

**主要功能**:
- 单独测试MSE/FGD/FSDlike蒸馏损失的最小训练脚本
- 参考warmup_training_v1.py的数据/日志风格
- 仅替换特征对齐损失
- 用于快速验证损失函数效果

**特点**:
- 1223行，完整训练脚本
- 支持多种损失函数（MSE、FGD、FSDlike）
- 单图过拟合测试

**状态**: ⚠️ 测试脚本，可归档

---

### 4. `unified_model_test.py` 📊 **评估脚本**
**功能**: 统一模型测试评估

**主要功能**:
- 在固定测试集上使用统一指标评估已训练模型
- 统一指标：Feature MSE/MAE、Cosine、Edge BCE+Dice
- 对多个checkpoint进行公平对比
- 复用训练脚本的关键逻辑（数据集类、模型组件、评估流程）

**特点**:
- 514行，完整评估脚本
- 支持多checkpoint对比
- 生成对比报告

**状态**: ⚠️ 测试脚本，可归档

---

### 5. `bbox/sa1b_bbox_extractor.py` 📦 **工具脚本**
**功能**: SA-1B实例框提取器

**主要功能**:
- 从SA-1B JSON标注文件快速提取实例框
- 提取2个大框+1个中框
- 支持NMS（非极大值抑制）
- 支持CUDA加速
- 可视化边界框

**特点**:
- 661行，完整工具
- 支持批量处理
- 可配置阈值参数

**状态**: ✅ 保留（有用工具）

---

## 🛠️ Fix目录（NPZ修复工具）

### 6. `fix/bulk_fix_npz_features.py`
**功能**: 批量替换NPZ中的P4/P5特征，并标记`feature_flag=1`

**特点**:
- 递归扫描指定目录
- 自动去重
- 原子写入

### 7. `fix/fix_npz_features_inplace.py`
**功能**: 原地修复NPZ特征（重算P4/P5），基于样本清单处理

**特点**:
- 支持NPZ文件分散存储
- 灵活的文件查找策略

### 8. `fix/fix_train_test_npz.py`
**功能**: 修复train_1200/test目录中extracted下的NPZ

**特点**:
- 针对特定目录结构优化
- 跨设备兼容

### 9. `fix/update_edge_maps_from_npz.py`
**功能**: 批量更新NPZ文件中的边缘图（使用Method B）

**特点**:
- 完全独立实现
- 支持多种过滤选项

**详细文档**: 见 `tools/core/fix/README.md`

---

## 🎯 Prompt目录（SAM2 Prompt工具）

### 10. `prompt/builder.py`
**功能**: Prompt构建器

**用途**: 构建SAM2的prompt输入（点、框、掩码等）

### 11. `prompt/prompts.py`
**功能**: Prompt工具函数

**用途**: Prompt相关的辅助函数

### 12. `prompt/seg_head.py`
**功能**: 分割头测试

**用途**: 测试SAM2的分割头功能

### 13. `prompt/test_*.py` (7个测试脚本)
- `test_large_model_standalone.py` - 大模型独立测试
- `test_npz_vs_realtime_features.py` - NPZ vs实时特征对比
- `test_npz_vs_realtime_segmentation.py` - NPZ vs实时分割对比
- `test_real_sam2.py` - 真实SAM2测试
- `test_seg_head.py` - 分割头测试
- `test_seg_head_with_reference.py` - 带参考的分割头测试
- `compare_base_large_256ch_features.py` - Base/Large模型256通道特征对比
- `compare_multi_models_features.py` - 多模型特征对比
- `diagnose_large_features.py` - 大特征诊断

**状态**: ⚠️ 全部为测试/诊断脚本，可归档

---

## 📊 分类统计

| 类别 | 脚本数 | 状态 |
|------|--------|------|
| **核心工具** | 1个 | ✅ 保留 |
| **有用工具** | 1个 (bbox) | ✅ 保留 |
| **修复工具** | 4个 | ✅ 保留（已在fix目录整理） |
| **实验/测试脚本** | 3个 | ⚠️ 可归档 |
| **Prompt工具** | 3个 | ✅ 保留 |
| **Prompt测试脚本** | 9个 | ⚠️ 可归档 |

---

## 🎯 建议操作

### 保留的脚本（核心 + 有用）
1. ✅ `extract_features_v1.py` - **核心工具，必须保留**
2. ✅ `bbox/sa1b_bbox_extractor.py` - 有用工具
3. ✅ `fix/*.py` - NPZ修复工具（4个，已整理）
4. ✅ `prompt/builder.py`, `prompts.py`, `seg_head.py` - Prompt核心工具（3个）

### 可归档的脚本（实验/测试）
1. ⚠️ `extract_features_edge_comparison.py` - 边缘提取方法对比实验
2. ⚠️ `train_distill_single_test.py` - 单图蒸馏训练测试
3. ⚠️ `unified_model_test.py` - 统一模型测试评估
4. ⚠️ `prompt/test_*.py` (9个) - 各种测试脚本

**建议归档目录**: `tools/archive/core/`

---

**最后更新**: 2025-11-05


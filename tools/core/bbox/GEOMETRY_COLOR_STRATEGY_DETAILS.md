# Geometry+Color 背景判定策略详细说明

## 一、基础计算

### 1.1 亮度计算
- **公式**: `L = 0.299 * R + 0.587 * G + 0.114 * B` (感知亮度，ITU-R BT.601)
- **用途**: 用于区分白天/夜间背景类型

### 1.2 位置计算
- **top_y**: `y / H` (bbox顶部距离图像顶部的归一化距离)
- **bottom_y**: `(y + h) / H` (bbox底部距离图像顶部的归一化距离)

### 1.3 形状特征
- **solidity**: `mask_area / convex_hull_area` (掩码面积 / 凸包面积)
- **hole_frac**: `internal_hole_area / (mask_area + internal_hole_area)` (内部孔洞面积占比)
- **aspect_ratio**: `w / h` (bbox宽高比)

---

## 二、特殊规则（最高优先级）

### 2.1 自动背景判定
**条件**: `solidity < 0.6` **且** `hole_frac > 0.15`
- **含义**: 太凹且孔洞太多，直接判定为背景（地面被大量物体阻挡）
- **返回**: 直接返回 `is_bg=True, bg_type='ground'`
- **优先级**: 最高，不检查颜色等其他条件

---

## 三、天空（SKY）判定

### 3.1 颜色条件（**不按亮度分支，同时判断所有条件**）
满足**任意一个**即通过：

#### 3.1.1 蓝色天空 (day_blue_sky)
- **条件**: `B > max(R, G) + 5`
- **含义**: 蓝色通道明显大于红色和绿色通道

#### 3.1.2 白色云朵 (day_white_cloud)
- **条件**: `max(R, G, B) - min(R, G, B) < 30` **且** `max(R, G, B) > 180`
- **含义**: RGB三个通道接近（差异<30）且至少有一个通道>180

#### 3.1.3 夜空 (night_sky)
- **条件1（纯黑色）**: `max(R, G, B) < 20`
- **条件2（偏蓝暗色）**: `B > R` **且** `B > G` **且** `B > 30`
- **最终**: `is_black_sky OR (B > R AND B > G AND B > 30)`

### 3.2 位置条件
- **条件**: `top_y <= 0.02`
- **含义**: bbox顶部距离图像顶部 ≤ 2%

### 3.3 面积条件
- **条件**: `area_ratio > 0.1`
- **含义**: 掩码面积 > 图像面积的10%

### 3.4 最终判定
- **通过条件**: `sky_color_ok AND sky_pos_ok AND sky_area_ok`
- **优先级**: 1（最高）

---

## 四、地面（GROUND）判定

### 4.1 颜色条件（**按亮度分支**）
满足**任意一个**即通过：

#### 4.1.1 亮混凝土 (day_bright_ground)
- **亮度要求**: `L >= 120`
- **颜色条件**: `min(R, G, B) >= 100` **且** `max(R, G, B) - min(R, G, B) < 20`
- **含义**: 整体偏白，RGB差异极小

#### 4.1.2 中等亮度水泥地 (mid_concrete)
- **亮度要求**: `95 <= L < 120`
- **颜色条件**: `min(R, G, B) >= 85` **且** `max(R, G, B) - min(R, G, B) < 15`
- **含义**: 略暗，但依旧接近无色

#### 4.1.3 夜间地面 (night_ground)
- **亮度要求**: `35 <= L <= 70`
- **颜色条件**: 
  - `min(R, G, B) >= 35`
  - `max(R, G, B) <= 110`
  - `max(R, G, B) - min(R, G, B) < 15`
- **含义**: 中等暗度，RGB差异小

### 4.2 位置条件
- **条件**: `bottom_y >= 0.98`
- **含义**: bbox底部距离图像顶部 ≥ 98%

### 4.3 面积条件
- **条件**: `area_ratio > 0.1`
- **含义**: 掩码面积 > 图像面积的10%

### 4.4 形状条件（调用 `evaluate_surface_shape`）
- **参数**: `require_wide=True`, `forbid_large_holes=False`
- **详细逻辑见第8节**

### 4.5 最终判定
- **通过条件**: `ground_color_ok AND ground_pos_ok AND ground_area_ok AND ground_shape_ok`
- **优先级**: 2

---

## 五、雪地（SNOW）判定

### 5.1 颜色条件
满足**任意一个**即通过：

#### 5.1.1 亮雪 (bright)
- **条件**: `R > 220` **且** `G > 220` **且** `B > 220`
- **含义**: 三个通道都>220

#### 5.1.2 浅雪 (light)
- **条件**: 
  - `180 <= R <= 255` **且** `180 <= G <= 255` **且** `180 <= B <= 255`
  - **且** `max(R, G, B) - min(R, G, B) < 30`
- **含义**: RGB都在180-255范围内，且差异<30

### 5.2 位置条件
- **条件**: `bottom_y >= 0.98`
- **含义**: bbox底部距离图像顶部 ≥ 98%

### 5.3 面积条件
- **条件**: `area_ratio > 0.1`
- **含义**: 掩码面积 > 图像面积的10%

### 5.4 最终判定
- **通过条件**: `is_snow AND snow_pos_ok AND snow_area_ok`
- **优先级**: 3
- **注意**: 雪地**不需要**形状判断

---

## 六、草地（GRASS）判定

### 6.1 颜色条件（**按亮度分支**）
满足**任意一个**即通过：

#### 6.1.1 白天草地 (day_grass)
- **亮度要求**: `L >= 105`
- **颜色条件**: `G >= 110` **且** `G > max(R, B) + 20`
- **含义**: 绿色通道明显大于红色和蓝色通道

#### 6.1.2 中等亮度草地 (mid_grass)
- **亮度要求**: `80 <= L < 105`
- **颜色条件**: `G >= 85` **且** `G > max(R, B) + 15`
- **含义**: 中等亮度的绿色

#### 6.1.3 绿化带 (lawn)
- **亮度要求**: `L >= 70`
- **颜色条件**: 
  - `G > 70`
  - `G > max(R, B) + 10`
  - `G > R + 5`
- **含义**: 更宽容的绿色条件（用于绿化带）

#### 6.1.4 夜间草地 (night_grass)
- **亮度要求**: `L <= 55`
- **颜色条件**: `G > R` **且** `G > B` **且** `G > 30`
- **含义**: 暗色但绿色通道仍最大

### 6.2 位置条件
- **条件**: **无要求**（`grass_pos_ok = True`）
- **含义**: 草地可以在图像任意位置

### 6.3 面积条件（**根据颜色类型使用不同阈值**）
- **普通草地**: `area_ratio > 0.1` (10%)
- **绿化带 (lawn)**: `area_ratio > 0.03` (3%)
- **最终**: 
  - `(is_day_grass AND area_ratio > 0.1) OR`
  - `(is_mid_grass AND area_ratio > 0.1) OR`
  - `(is_night_grass AND area_ratio > 0.1) OR`
  - `(is_lawn AND area_ratio > 0.03)`

### 6.4 形状条件
- **条件**: **不需要形状判断**
- **含义**: 草地只依赖颜色和面积

### 6.5 最终判定
- **通过条件**: `is_green AND grass_area_ok_final`
- **优先级**: 5（最低）

---

## 七、水面（WATER）判定

### 7.1 颜色条件（**按亮度分支**）
满足**任意一个**即通过：

#### 7.1.1 透明水 (transparent_water)
- **亮度要求**: `L >= 120`
- **颜色条件**: 
  - `max(R, G, B) - min(R, G, B) < 20`
  - `max(R, G, B) >= 150`
  - `B >= G - 10` **且** `B >= R - 10`
  - `bottom_y >= 0.98`
- **含义**: 很亮，RGB接近，偏蓝，且从底部开始

#### 7.1.2 绿色水 (green_water)
- **亮度要求**: `L >= 120`
- **颜色条件**: `G >= 110` **且** `G > max(R, B) + 20`
- **含义**: 很亮，绿色通道明显最大

#### 7.1.3 蓝水 (blue_water)
- **分支1（高亮度）**: `L >= 120` **且** `B >= 120` **且** `B >= max(R, G) + 10`
- **分支2（中亮度）**: `90 <= L < 120` **且** `B >= 105` **且** `B >= max(R, G) + 10`
- **最终**: `blue_cond_high OR blue_cond_mid`

#### 7.1.4 偏绿水 (greenish_water)
- **亮度要求**: `90 <= L < 120`
- **颜色条件**: 
  - `G >= 95`
  - `abs(G - B) <= 20`
  - `G > R + 5`
  - `max(R, G, B) - min(R, G, B) < 30`
  - `bottom_y >= 0.98`
- **含义**: 中等亮度，绿色和蓝色接近，且从底部开始

#### 7.1.5 低亮度绿水 (low_green_water)
- **亮度要求**: `60 <= L < 90`
- **颜色条件**: 
  - `G > B + 5`
  - `abs(B - R) <= 10`
  - `max(R, G, B) - min(R, G, B) < 25`
  - `bottom_y >= 0.98`
- **含义**: 中等暗度，绿色略大于蓝色，且从底部开始

#### 7.1.6 夜间水 (night_water)
- **亮度要求**: `L <= 60`
- **颜色条件**: 
  - `B >= 35`
  - `B > R + 5`
  - `B > G + 5`
  - `max(R, G, B) < 85`
- **含义**: 很暗，蓝色通道最大

#### 7.1.7 黑水 (black_water)
- **亮度要求**: `L <= 45`
- **颜色条件**: 
  - `max(R, G, B) < 60`
  - `bottom_y >= 0.98`
- **含义**: 非常暗，且从底部开始

#### 7.1.8 暗水 (dark_water)
- **亮度要求**: `L <= 80`
- **颜色条件**: 
  - `20 <= R <= 80` **且** `20 <= G <= 80` **且** `20 <= B <= 80`
  - `B >= G - 5` **且** `B >= R - 5`
- **含义**: 中等暗度，RGB都在20-80范围内，蓝色略大

#### 7.1.9 灰水 (gray_water)
- **亮度要求**: `L >= 100`
- **颜色条件**: 
  - `max(R, G, B) - min(R, G, B) < 15`
  - `bottom_y >= 0.98`
- **含义**: 较亮，RGB差异很小，且从底部开始

### 7.2 位置条件
- **条件**: `bottom_y >= 0.98`
- **含义**: bbox底部距离图像顶部 ≥ 98%
- **注意**: 部分颜色子类型（如transparent_water, greenish_water, low_green_water, black_water, gray_water）在颜色条件中已经包含了位置要求

### 7.3 面积条件
- **条件**: `area_ratio > 0.1`
- **含义**: 掩码面积 > 图像面积的10%

### 7.4 形状条件
- **条件**: **不需要形状判断**
- **含义**: 水面只依赖颜色、面积和位置

### 7.5 最终判定
- **通过条件**: `water_color_ok AND water_area_ok AND water_pos_ok`
- **优先级**: 4

---

## 八、形状判定函数 (`evaluate_surface_shape`)

### 8.1 输入参数
- **solidity**: 掩码面积 / 凸包面积
- **hole_frac**: 内部孔洞面积占比
- **aspect_ratio**: bbox宽高比
- **require_wide**: 是否要求横向展开（默认True）
- **forbid_large_holes**: 是否禁止大孔洞（默认False）

### 8.2 中间计算
- **concavity**: `1.0 - solidity` (凹陷程度)
- **wide**: `aspect_ratio > 1.4` (横向展开)
- **has_hole**: `hole_frac > 0.12` (有孔洞)
- **large_hole**: `hole_frac > 0.18` (大孔洞)
- **very_concave**: `concavity >= 0.35` (即 `solidity <= 0.65`)
- **moderate_concave**: `concavity >= 0.2` (即 `solidity <= 0.8`)

### 8.3 判定逻辑（按顺序）

#### 8.3.1 禁止大孔洞
- **条件**: `forbid_large_holes == True` **且** `large_hole == True`
- **结果**: **FAIL** (`reason='large_hole_forbidden'`)

#### 8.3.2 太凸
- **条件**: `solidity > 0.9`
- **结果**: **FAIL** (`reason='too_convex'`)

#### 8.3.3 明显凹陷
- **条件**: `very_concave == True` **且** (`require_wide == False` **或** `wide == True`)
- **结果**: **PASS** (`reason='strong_concavity'`)

#### 8.3.4 中等凹陷（有支持）
- **条件**: `moderate_concave == True` **且** ((`require_wide == True` **且** `wide == True`) **或** `has_hole == True`)
- **结果**: **PASS** (`reason='moderate_concavity_supported'`)

#### 8.3.5 中等凹陷（不要求横向）
- **条件**: `moderate_concave == True` **且** `require_wide == False`
- **结果**: **PASS** (`reason='moderate_concavity'`)

#### 8.3.6 孔洞+横向
- **条件**: `has_hole == True` **且** `wide == True`
- **结果**: **PASS** (`reason='hole_plus_wide'`)

#### 8.3.7 其他情况
- **结果**: **FAIL** (`reason='not_enough_concavity'`)

### 8.4 使用场景
- **地面 (ground)**: `require_wide=True`, `forbid_large_holes=False`
- **草地 (grass)**: **不使用形状判断**
- **水面 (water)**: **不使用形状判断**

---

## 九、判定优先级

按以下顺序检查，**第一个匹配的即返回**：

1. **特殊规则**: `solidity < 0.6 AND hole_frac > 0.15` → `ground`
2. **天空**: `sky_color_ok AND sky_pos_ok AND sky_area_ok` → `sky`
3. **地面**: `ground_color_ok AND ground_pos_ok AND ground_area_ok AND ground_shape_ok` → `ground`
4. **雪地**: `is_snow AND snow_pos_ok AND snow_area_ok` → `snow`
5. **水面**: `water_color_ok AND water_area_ok AND water_pos_ok` → `water`
6. **草地**: `is_green AND grass_area_ok_final` → `grass`
7. **其他**: 返回 `is_bg=False, bg_type='unknown'`

---

## 十、潜在问题分析

### 10.1 天空判定
- ✅ **已修复**: 去掉了亮度分支限制，现在同时判断所有颜色条件
- ⚠️ **潜在问题**: 
  - `day_blue_sky` 条件 `B > max(R, G) + 5` 可能过于宽松，可能误判某些偏蓝的前景
  - `day_white_cloud` 的 `max_RGB > 180` 可能遗漏较暗的云

### 10.2 地面判定
- ⚠️ **潜在问题**: 
  - `night_ground` 的亮度范围 `35 <= L <= 70` 可能与某些前景重叠
  - 形状判定要求 `solidity <= 0.65` 或配合孔洞/横向，可能遗漏某些平整地面

### 10.3 雪地判定
- ✅ **简单明确**: 只依赖颜色、位置、面积
- ⚠️ **潜在问题**: 
  - `light` 条件范围较宽（180-255），可能与某些浅色前景重叠

### 10.4 草地判定
- ⚠️ **潜在问题**: 
  - `lawn` 条件 `G > 70 AND G > max(R, B) + 10 AND G > R + 5` 可能过于宽松
  - 不同亮度分支的条件可能有重叠，导致同一颜色可能同时满足多个条件

### 10.5 水面判定
- ⚠️ **潜在问题**: 
  - 颜色子类型过多（9种），可能有重叠和冲突
  - `dark_water` 条件 `20 <= R,G,B <= 80` 范围较宽，可能误判
  - `gray_water` 条件 `diff < 15 AND bottom_y >= 0.98` 可能遗漏某些灰色前景
  - 部分颜色子类型（如 `transparent_water`, `greenish_water`）在颜色条件中已经包含位置要求，但最终还要检查 `water_pos_ok`，可能导致重复检查

### 10.6 形状判定
- ⚠️ **潜在问题**: 
  - `evaluate_surface_shape` 的逻辑较复杂，多个条件分支可能导致难以调试
  - `very_concave` 阈值 `solidity <= 0.65` 可能过于严格，某些地面可能无法通过

### 10.7 特殊规则
- ✅ **明确**: `solidity < 0.6 AND hole_frac > 0.15` 直接判定为背景
- ⚠️ **潜在问题**: 这个规则优先级最高，可能在某些情况下过于激进

---

## 十一、建议改进方向

1. **简化水面颜色判定**: 合并某些相似的颜色子类型，减少重叠
2. **统一位置要求**: 对于在颜色条件中已包含位置要求的子类型，避免重复检查
3. **优化形状判定**: 考虑简化 `evaluate_surface_shape` 的逻辑，或添加更多调试信息
4. **添加调试模式**: 输出每个判定步骤的详细结果，便于排查问题
5. **考虑颜色空间**: 当前使用RGB，可考虑使用HSV或LAB颜色空间，可能更符合人类感知


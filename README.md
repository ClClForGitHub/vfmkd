# VFMKD - 视觉基础模型蒸馏

一个支持多任务（检测+分割）、多backbone（YOLOv8/ViT/Mamba/RepViT）、多教师（SAM/DINO）的知识蒸馏框架。

## 项目特点

- **多任务支持**: 同时支持目标检测和图像分割
- **通用Backbone接口**: 支持YOLOv8、ViT、Mamba、RepViT等多种backbone
- **多教师蒸馏**: 支持SAM、DINO等教师模型
- **离线+在线混合**: 主要使用离线特征，部分使用在线特征
- **模块化设计**: 高度模块化，易于扩展

## 项目结构

```
VFMKD/
├── vfmkd/                          # 核心代码库
│   ├── models/                     # 模型实现
│   │   ├── backbones/              # Backbone实现
│   │   ├── heads/                  # 检测/分割头
│   │   ├── necks/                  # 特征融合网络
│   │   └── unified_model.py        # 统一模型接口
│   ├── teachers/                   # 教师模型
│   │   ├── sam_teacher.py          # SAM教师
│   │   ├── sam2_teacher.py         # SAM2教师
│   │   └── dino_teacher.py         # DINO教师
│   ├── distillation/               # 蒸馏实现
│   │   ├── offline_distiller.py   # 离线蒸馏
│   │   ├── online_distiller.py     # 在线蒸馏
│   │   └── losses/                 # 损失函数
│   ├── datasets/                   # 数据集加载器
│   ├── utils/                      # 工具函数
│   └── core/                       # 训练/评估核心
├── configs/                        # 配置文件
│   ├── backbones/                  # Backbone配置
│   ├── teachers/                   # 教师模型配置
│   └── experiments/                # 实验配置
├── datasets/                       # 数据集目录
│   └── coco128/                    # COCO128数据集
├── weights/                        # 预训练权重
├── teacher_features/               # 教师特征缓存
├── tools/                         # 工具脚本
├── scripts/                       # 环境设置脚本
└── requirements.txt               # 依赖包
```

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd VFMKD

# 设置环境
bash scripts/setup_env.sh

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 下载SAM权重
python tools/download_sam_weights.py --model vit_h

# 下载其他模型权重
python tools/download_models.py
```

### 3. 准备数据集

```bash
# 下载数据集
bash scripts/download_datasets.sh

# 或使用内置的coco128数据集
# 数据集已位于 datasets/coco128/
```

### 4. 预计算教师特征

```bash
# 使用SAM2教师提取特征
python tools/prepare_teacher_features.py \
    --teacher sam2 \
    --dataset coco128 \
    --output teacher_features/sam2/
```

### 5. 开始训练

```bash
# 使用YOLOv8 backbone + SAM2教师进行蒸馏训练
python tools/train.py \
    --config configs/experiments/yolov8_sam2_coco.yaml \
    --backbone yolov8 \
    --teacher sam2 \
    --dataset coco128
```

## 支持的模型

### Backbone
- **YOLOv8**: CSPDarknet架构，多尺度特征输出
- **ViT**: Vision Transformer，支持不同尺寸
- **Mamba**: 基于状态空间模型的高效架构
- **RepViT**: 从EdgeSAM迁移的高效CNN架构

### 教师模型
- **SAM**: Segment Anything Model，强大的分割能力
- **SAM2**: SAM2.1 Hiera，更先进的分割模型
- **DINO**: 自监督学习模型，强大的特征表示

### 任务
- **目标检测**: 基于YOLO和DETR的检测头
- **图像分割**: 基于SAM的分割头

## 配置示例

### YOLOv8 + SAM2 蒸馏配置

```yaml
# configs/experiments/yolov8_sam2_coco.yaml
experiment_name: yolov8_sam2_coco_detection

model:
  backbone:
    type: yolov8
    config: configs/backbones/yolov8.yaml
  heads:
    detection:
      type: yolo_head
      num_classes: 80

teachers:
  sam2:
    enabled: true
    offline: true
    model_path: weights/sam2.1_hiera_base_plus.pt
    feature_path: teacher_features/sam2/

distillation:
  offline_ratio: 0.8
  losses:
    feature_mse:
      weight: 1.0
      teacher: sam2
    detection:
      weight: 5.0

data:
  dataset: coco128
  batch_size: 16
  num_workers: 8

train:
  epochs: 100
  lr: 0.01
  optimizer: adamw
```

## 核心特性

### 1. 多尺度特征蒸馏
- 支持P3/P4/P5多尺度特征对齐
- 自动特征维度匹配
- 空间尺寸插值对齐

### 2. 离线+在线混合蒸馏
- 80%使用预计算特征（离线）
- 20%使用实时特征（在线）
- 平衡效率和效果

### 3. 多教师支持
- 同时使用多个教师模型
- 不同教师使用不同损失函数
- 可配置的损失权重

### 4. 模块化设计
- 统一的Backbone接口
- 可插拔的Head设计
- 灵活的配置系统

## 开发指南

### 添加新的Backbone

1. 继承`BaseBackbone`类
2. 实现`forward`、`get_feature_dims`、`get_feature_strides`方法
3. 在`configs/backbones/`中添加配置文件
4. 更新`vfmkd/models/backbones/__init__.py`

### 添加新的教师模型

1. 继承`BaseTeacher`类
2. 实现特征提取和掩码预测方法
3. 在`configs/teachers/`中添加配置文件
4. 更新`vfmkd/teachers/__init__.py`

### 添加新的损失函数

1. 在`vfmkd/distillation/losses/`中实现损失函数
2. 继承相应的基类
3. 在蒸馏器中注册新的损失

## 性能优化

- 使用混合精度训练
- 支持分布式训练
- 特征缓存机制
- 内存优化策略

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM): 特征存储和蒸馏流程参考
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLOv8实现参考
- [SAM2](https://github.com/facebookresearch/segment-anything-2): SAM2模型实现
- [DINO](https://github.com/facebookresearch/dino): DINO教师模型参考
# VFMKD 快速开始

## 🚀 三步上传到GitHub

### 步骤1: 创建GitHub仓库
访问 https://github.com/new 创建新仓库
- 仓库名: `vfmkd`
- 可见性: Public（推荐）
- **不要**勾选"Initialize with README"

### 步骤2: 连接远程仓库
```bash
cd C:\AiBuild\paper\detect\EdgeSAM-master\VFMKD
git remote add origin https://github.com/YOUR_USERNAME/vfmkd.git
```

### 步骤3: 推送代码
```bash
git push -u origin main
```

✅ 完成！访问您的GitHub仓库查看代码。

---

## 📦 用户使用指南

### 克隆并安装

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/vfmkd.git
cd vfmkd

# 2. 创建虚拟环境（推荐）
conda create -n vfmkd python=3.10
conda activate vfmkd

# 3. 安装依赖
pip install -r requirements.txt
```

### 下载权重

```bash
# SAM2权重
python tools/download_sam_weights.py

# YOLOv8权重
python tools/download_yolov8_weights.py
```

### 准备数据

```bash
# 下载COCO128（示例数据集）
bash scripts/download_datasets.sh

# 或使用自己的数据集，放置在 datasets/ 目录
```

### 开始训练

```bash
# 使用示例配置
python tools/train.py --config configs/experiments/example.yaml

# 或自定义配置
python tools/train.py \
    --backbone yolov8 \
    --teacher sam2 \
    --dataset coco128 \
    --epochs 100
```

---

## 📊 项目统计

| 项目 | 数值 |
|------|------|
| 提交文件数 | 209个 |
| 代码行数 | 38,213行 |
| Python文件 | 155个 |
| 配置文件 | 20个 |
| 文档文件 | 17个 |
| 仓库大小 | ~5-10 MB |

---

## 🔍 目录说明

```
VFMKD/
├── vfmkd/              # 核心代码
│   ├── models/         # 模型（backbones、heads、necks）
│   ├── distillation/   # 蒸馏损失函数
│   ├── teachers/       # 教师模型
│   └── sam2/          # SAM2集成
├── configs/           # 配置文件
├── tools/             # 训练/评估/可视化工具
├── tests/             # 单元测试
├── docs/              # 文档
├── scripts/           # Shell脚本
├── requirements.txt   # Python依赖
└── setup.py          # 安装脚本
```

---

## 🛠️ 常用命令

### 训练
```bash
# 基础训练
python tools/train.py --config CONFIG_FILE

# 使用预训练权重
python tools/train.py --config CONFIG_FILE --pretrained weights/model.pth

# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config CONFIG_FILE
```

### 评估
```bash
python tools/eval.py --config CONFIG_FILE --checkpoint PATH_TO_CHECKPOINT
```

### 可视化
```bash
# 可视化backbone特征
python tools/vis_backbone_features.py --image IMAGE_PATH

# 可视化SAM2结果
python tools/vis_sam2_reference.py --image IMAGE_PATH
```

---

## 💡 配置示例

创建自定义配置文件 `configs/experiments/my_experiment.yaml`:

```yaml
experiment_name: my_yolov8_sam2_experiment

model:
  backbone:
    type: yolov8
    model: yolov8s
  heads:
    detection:
      type: yolo_head
      num_classes: 80

teachers:
  sam2:
    enabled: true
    model_path: weights/sam2.1_hiera_base_plus.pt

data:
  dataset: coco128
  batch_size: 16
  num_workers: 4

train:
  epochs: 100
  lr: 0.01
  optimizer: adamw

distillation:
  losses:
    feature_mse:
      weight: 1.0
```

---

## 📚 更多资源

- 📖 完整文档: [README.md](README.md)
- 🔧 部署指南: [GITHUB_DEPLOY.md](GITHUB_DEPLOY.md)
- 📝 贡献指南: [CONTRIBUTING.md](CONTRIBUTING.md)
- 📊 部署总结: [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)

---

## 🆘 遇到问题？

1. **检查日志**: 训练日志通常保存在 `outputs/` 目录
2. **查看文档**: 阅读 `docs/` 目录下的详细文档
3. **提交Issue**: 在GitHub仓库创建Issue
4. **加入讨论**: 参与GitHub Discussions

---

## 📞 联系方式

- GitHub Issues: https://github.com/YOUR_USERNAME/vfmkd/issues
- Email: vfmkd@example.com

---

**Happy Coding! 🎉**


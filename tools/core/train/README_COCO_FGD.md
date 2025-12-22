# COCO 微调对比实验脚本

该目录提供 **单一脚本** `train_coco_fgd_auto.sh`，用于在 COCO2017 数据集上比较“加载FGD蒸馏backbone”与“随机初始化backbone”的检测微调效果。两个实验除“是否加载预训练权重”和“GPU设备”外，其余超参保持完全一致，便于公平对比。

---

## 1. 脚本概览

- **脚本路径**：`tools/core/train/train_coco_fgd_auto.sh`
- **核心训练文件**：`tools/core/train/train_coco_mmdet_lego.py`
- **默认蒸馏权重**：`outputs/distill_single_test_FGD/.../best_backbone_mmdet.pth`
- **数据集**：`/home/team/zouzhiyuan/dataset/COCO2017`
- **实验输出**：`./work_dirs/coco_finetune_compare_pretrained` 与 `./work_dirs/coco_finetune_compare_random`

脚本启动后可运行以下模式：

| 环境变量 `RUN_MODE` | 行为 |
| --- | --- |
| `both` (默认) | 先运行“加载预训练”实验，再运行“随机初始化”实验 |
| `pretrained` | 只运行加载预训练的实验 |
| `random` | 只运行随机初始化的实验 |

---

## 2. 快速开始

```bash
cd /home/team/zouzhiyuan/vfmkd
bash tools/core/train/train_coco_fgd_auto.sh
```

默认配置：
- Batch Size = 32 / GPU
- 冻结Backbone前 50 个 epoch，之后解冻
- 预训练实验使用 GPU `5`，随机初始化使用 GPU `6`
- 训练 100 epoch（由 `train_coco_mmdet_lego.py` 控制）

---

## 3. 自定义配置

通过环境变量即可修改所有关键参数：

```bash
# 仅运行加载预训练的实验，使用 GPU 1、Batch Size 16
RUN_MODE=pretrained PRETRAINED_GPUS=1 BATCH_SIZE=16 \
    bash tools/core/train/train_coco_fgd_auto.sh

# 运行对比实验，指定两个实验使用不同的物理 GPU
PRETRAINED_GPUS=4 RANDOM_GPUS=5 bash tools/core/train/train_coco_fgd_auto.sh

# 指定自定义蒸馏权重路径
PRETRAINED_BACKBONE=/path/to/backbone.pth bash tools/core/train/train_coco_fgd_auto.sh

# 使用相同 GPU，同步运行两个实验（按顺序执行）
PRETRAINED_GPUS=7 RANDOM_GPUS=7 bash tools/core/train/train_coco_fgd_auto.sh
```

### 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_MODE` | `both` | 运行模式：`both/pretrained/random` |
| `PRETRAINED_BACKBONE` | 默认best模型 | 蒸馏backbone路径 |
| `PRETRAINED_GPUS` | `5` | 预训练实验使用的物理 GPU ID（如 `1,2`） |
| `RANDOM_GPUS` | `6` | 随机实验使用的物理 GPU ID |
| `BATCH_SIZE` | `32` | 每张 GPU 的 batch size |
| `FREEZE_BACKBONE` | `true` | 是否冻结 backbone（仅恢复 Neck/Head） |
| `UNFREEZE_EPOCH` | `50` | `FREEZE_BACKBONE=true` 时，在哪个 epoch 解冻 |
| `WORK_DIR_BASE` | `./work_dirs/coco_finetune_compare` | 输出目录前缀 |

> **提示**：如需在后台运行，可结合 `nohup`：
> ```bash
> nohup RUN_MODE=both PRETRAINED_GPUS=4 RANDOM_GPUS=5 \
>     bash tools/core/train/train_coco_fgd_auto.sh > coco_compare.log 2>&1 &
> ```

---

## 4. 训练逻辑说明

- **共用脚本** `train_coco_mmdet_lego.py` 设置了：
  - SGD 优化器：lr=0.01，momentum=0.937，weight_decay=0.0005
  - LR 调度：3 个 epoch warmup + Cosine 退火至 0.0001
  - 训练 100 epoch，默认 Batch Size = `--bs`
  - 数据路径写死为 `/home/team/zouzhiyuan/dataset/COCO2017`
- **随机初始化** 通过 `--random-init` 启用，脚本将跳过 checkpoint 加载
- **预训练实验** 通过 `--distilled-backbone <path>` 指定蒸馏权重
- 其余参数（Batch Size、冻结策略、runtime hook 等）完全一致

---

## 5. 输出目录结构

```
work_dirs/
├── coco_finetune_compare_pretrained/
│   ├── logs/
│   ├── latest.pth / epoch_x.pth
│   └── ...
└── coco_finetune_compare_random/
    ├── logs/
    ├── latest.pth / epoch_x.pth
    └── ...
```

---

## 6. 故障排查

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 找不到模型路径 | `PRETRAINED_BACKBONE` 配置错误 | 确认路径是否存在，并指向 `*_backbone_mmdet.pth` |
| CUDA OOM | Batch Size 过大 | 使用 `BATCH_SIZE=16` 等较小值运行 |
| 数据集路径错误 | COCO 数据未放到默认位置 | 确保 `/home/team/zouzhiyuan/dataset/COCO2017` 可访问 |
| RUN_MODE 无效 | 传入非法值 | 只能为 `both` / `pretrained` / `random` |

---

## 7. 常见用法速查

| 目的 | 命令示例 |
| --- | --- |
| 运行完整对比实验（默认GPU） | `bash train_coco_fgd_auto.sh` |
| 只跑随机初始化实验 | `RUN_MODE=random bash train_coco_fgd_auto.sh` |
| 只跑预训练实验，使用 GPU 1 | `RUN_MODE=pretrained PRETRAINED_GPUS=1 bash train_coco_fgd_auto.sh` |
| 使用自定义蒸馏权重 | `PRETRAINED_BACKBONE=/path/to/model.pth bash train_coco_fgd_auto.sh` |
| 相同 GPU 顺序跑两次 | `PRETRAINED_GPUS=4 RANDOM_GPUS=4 bash train_coco_fgd_auto.sh` |

如需进一步定制（例如修改训练 epoch、学习率、数据增强等），请直接编辑 `train_coco_mmdet_lego.py`。












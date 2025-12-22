# 实验启动脚本说明

本目录包含正式实验和测试脚本，用于 MSE、FGD 蒸馏实验以及各种测试场景。

## 📋 脚本列表

### 正式实验脚本

#### 1. `run_mse_experiment.sh` - MSE 蒸馏实验
- **GPU设备**: CUDA:5
- **Loss类型**: MSE
- **特征增强**: 无
- **默认配置**: 标准 MSE 损失，无边缘增强
- **Epochs**: 500
- **Batch Size**: 42

#### 2. `run_fgd_experiment.sh` - FGD 蒸馏实验
- **GPU设备**: CUDA:6
- **Loss类型**: FGD (Feature Guided Distillation)
- **特征增强**: 启用边缘增强 (`--enable-edge-boost`)
- **默认配置**: FGD 损失 + 边缘增强
- **Epochs**: 500
- **Batch Size**: 42

### 测试脚本

#### 3. `test_fgd_only.sh` - FGD 纯测试脚本
- **GPU设备**: CUDA:5
- **Loss类型**: FGD
- **特征增强**: 启用边缘增强
- **特点**: **仅测试 FGD 损失，不开启边缘任务和掩码任务**
- **用途**: 专门测试 FGD 损失函数本身的效果
- **Epochs**: 50（测试用）
- **Batch Size**: 42

#### 4. `test_mask_task.sh` - 辨析训练测试脚本
- **GPU设备**: CUDA:6
- **Loss类型**: MSE（基础损失）
- **特点**: 
  - 使用 **2000 张图像**进行快速测试
  - **10 epoch 后启动辨析训练**（掩码任务）
  - 5 epoch 启动边缘任务
  - 15 epoch 解冻掩码头
- **用途**: 专门测试辨析训练（掩码任务）的效果
- **Epochs**: 20
- **Batch Size**: 42
- **Max Images**: 2000

## 🚀 快速启动

### 方法1: 直接执行（推荐）

```bash
# 正式实验
cd /home/team/zouzhiyuan/vfmkd/tools/core/exper

# 启动 MSE 实验（GPU 5）
./run_mse_experiment.sh

# 启动 FGD 实验（GPU 6）
./run_fgd_experiment.sh

# 测试脚本
# FGD 纯测试（仅FGD损失，无边缘/掩码任务）
./test_fgd_only.sh

# 辨析训练测试（2000张图像，10 epoch后启动掩码任务）
./test_mask_task.sh
```

### 方法2: 后台运行（适合长时间训练）

```bash
# MSE 实验后台运行
nohup ./run_mse_experiment.sh > /dev/null 2>&1 &

# FGD 实验后台运行
nohup ./run_fgd_experiment.sh > /dev/null 2>&1 &
```

### 方法3: 使用 screen/tmux（推荐用于长时间训练）

```bash
# 使用 screen
screen -S mse_experiment
./run_mse_experiment.sh
# 按 Ctrl+A 然后 D 分离会话

screen -S fgd_experiment
./run_fgd_experiment.sh
# 按 Ctrl+A 然后 D 分离会话

# 重新连接
screen -r mse_experiment
screen -r fgd_experiment
```

## ⚙️ 配置说明

### 共同配置

两个脚本使用相同的训练配置（除了损失类型和特征增强）：

- **环境**: SSH
- **数据格式**: tar_shard（流式读取，保护机械硬盘IO）
- **Backbone**: YOLOv8
- **Epochs**: 50
- **Batch Size**: 32（SSH环境自动优化）
- **Learning Rate**: 1e-3
- **总图像数**: 109960（用于进度条显示）
- **数据目录**: `/home/team/zouzhiyuan/dataset/sa1b_tar_shards`
- **损失权重**: 
  - `feat_weight`: 1.0
  - `edge_weight`: 1.0
- **任务启动epoch**:
  - `edge_task_start_epoch`: 5
  - `mask_task_start_epoch`: 10

### MSE 实验特有配置

- **Loss Type**: `mse`
- **特征增强**: 无
- **运行标签**: `mse_gpu5`

### FGD 实验特有配置

- **Loss Type**: `fgd`
- **特征增强**: 启用 (`--enable-edge-boost`)
- **FGD 超参数**:
  - `fgd_alpha_fg`: 0.001（前景权重）
  - `fgd_beta_bg`: 0.0005（背景权重，前景的一半）
  - `fgd_alpha_edge`: 0.002（边缘权重，前景的两倍）
  - `fgd_temperature`: 1.0
- **运行标签**: `fgd_gpu6_edge_boost`

## 📁 输出目录结构

训练结果保存在：

```
/home/team/zouzhiyuan/vfmkd/outputs/
├── distill_single_test_MSE/
│   └── {timestamp}_yolov8_mse_gpu5/
│       ├── models/
│       │   ├── epoch_*_model.pth          # 完整checkpoint
│       │   ├── epoch_*_backbone_mmdet.pth # MMDet兼容backbone
│       │   ├── best_model.pth            # 最佳模型
│       │   └── best_backbone_mmdet.pth   # 最佳MMDet兼容backbone
│       └── visualizations/
└── distill_single_test_FGD/
    └── {timestamp}_yolov8_fgd_gpu6_edge_boost/
        └── ...
```

## 📊 日志文件

日志文件保存在：

```
/home/team/zouzhiyuan/vfmkd/tools/core/logs/
├── mse_experiment_{timestamp}.log
└── fgd_experiment_{timestamp}.log
```

日志文件包含：
- 完整的训练输出
- 每个epoch的loss统计
- 性能分析（数据加载时间、GPU利用率等）
- 错误信息（如果有）

## 🔧 自定义配置

如果需要修改配置，可以直接编辑脚本文件中的变量：

```bash
# 修改训练轮数
EPOCHS=100

# 修改学习率
LEARNING_RATE=5e-4

# 修改batch size
BATCH_SIZE=16

# 修改FGD参数（仅FGD脚本）
FGD_ALPHA_FG=0.002
```

## 📈 监控训练进度

### 查看实时日志

```bash
# 查看MSE实验日志
tail -f /home/team/zouzhiyuan/vfmkd/tools/core/logs/mse_experiment_*.log

# 查看FGD实验日志
tail -f /home/team/zouzhiyuan/vfmkd/tools/core/logs/fgd_experiment_*.log
```

### 查看GPU使用情况

```bash
# 查看GPU 5和GPU 6的使用情况
watch -n 1 nvidia-smi
```

### 查看训练输出目录

```bash
# 查看MSE实验输出
ls -lh /home/team/zouzhiyuan/vfmkd/outputs/distill_single_test_MSE/

# 查看FGD实验输出
ls -lh /home/team/zouzhiyuan/vfmkd/outputs/distill_single_test_FGD/
```

## ⚠️ 注意事项

1. **GPU资源**: 确保GPU 5和GPU 6可用，且没有被其他进程占用
2. **数据路径**: 确保数据目录 `/home/team/zouzhiyuan/dataset/sa1b_tar_shards` 存在且包含tar文件
3. **磁盘空间**: 确保有足够的磁盘空间保存模型checkpoint和日志
4. **训练时间**: 50个epoch可能需要数小时，建议使用screen/tmux或nohup后台运行
5. **进度条**: 进度条基于总图像数109960计算，如果实际数量不同，进度条可能略有偏差但不影响训练

## 🐛 故障排除

### 脚本无法执行

```bash
# 检查执行权限
ls -l run_*.sh

# 如果没有执行权限，添加权限
chmod +x run_*.sh
```

### GPU不可用

```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA设备数量
python -c "import torch; print(torch.cuda.device_count())"
```

### 数据路径错误

```bash
# 检查数据目录是否存在
ls -lh /home/team/zouzhiyuan/dataset/sa1b_tar_shards/

# 检查tar文件数量
ls /home/team/zouzhiyuan/dataset/sa1b_tar_shards/*.tar* | wc -l
```

## 📝 实验对比

两个实验的唯一区别：

| 配置项 | MSE实验 | FGD实验 |
|--------|---------|---------|
| GPU设备 | CUDA:5 | CUDA:6 |
| Loss类型 | MSE | FGD |
| 边缘增强 | ❌ | ✅ |
| FGD超参数 | N/A | 已配置 |

其他所有配置（epochs、batch size、learning rate等）完全相同，便于公平对比。


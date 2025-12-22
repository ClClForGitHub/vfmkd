#!/bin/bash
# ============================================================================
# MSE 蒸馏实验启动脚本
# GPU: 5
# Loss Type: MSE
# ============================================================================

set -e  # 遇到错误立即退出

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_distill_single_test.py"

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# ============================================================================
# 实验配置
# ============================================================================

# 基础配置
ENV_TYPE="ssh"
DATA_FORMAT="binary"  # 使用高性能二进制格式
LOSS_TYPE="mse"
BACKBONE="yolov8"
CUDA_DEVICE=4

# 训练参数
EPOCHS=500
BATCH_SIZE=24  # 降低到24以避免OOM
LEARNING_RATE=1e-3
TOTAL_IMAGES=109960  # 数据集总图像数（用于进度条显示）

# 损失权重
FEAT_WEIGHT=1.0
EDGE_WEIGHT=1.0

# 任务启动epoch
EDGE_TASK_START_EPOCH=5
MASK_TASK_START_EPOCH=999999  # 暂时禁用辨析头，等测试完成后再启用

# 数据路径（二进制数据集路径）
BINARY_DATA_PATH="/home/team/zouzhiyuan/dataset/sa1b_binary"

# 输出目录标签（用于区分实验）
RUN_TAG="mse_gpu4"

# ============================================================================
# 日志配置
# ============================================================================

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mse_experiment_${TIMESTAMP}.log"

# ============================================================================
# 启动训练
# ============================================================================

echo "============================================================================"
echo "MSE 蒸馏实验启动"
echo "============================================================================"
echo "GPU设备:        CUDA:${CUDA_DEVICE}"
echo "Loss类型:       ${LOSS_TYPE}"
echo "Backbone:       ${BACKBONE}"
echo "Epochs:         ${EPOCHS}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Learning Rate:  ${LEARNING_RATE}"
echo "总图像数:       ${TOTAL_IMAGES}"
echo "数据格式:       ${DATA_FORMAT}"
echo "数据目录:       ${BINARY_DATA_PATH}"
echo "运行标签:       ${RUN_TAG}"
echo "日志文件:       ${LOG_FILE}"
echo "============================================================================"
echo ""

# 执行训练命令
python "$TRAIN_SCRIPT" \
    --env "$ENV_TYPE" \
    --data-format "$DATA_FORMAT" \
    --binary-data-path "$BINARY_DATA_PATH" \
    --loss-type "$LOSS_TYPE" \
    --backbone "$BACKBONE" \
    --cuda-device "$CUDA_DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --feat-weight "$FEAT_WEIGHT" \
    --edge-weight "$EDGE_WEIGHT" \
    --total-images "$TOTAL_IMAGES" \
    --edge-task-start-epoch "$EDGE_TASK_START_EPOCH" \
    --mask-task-start-epoch "$MASK_TASK_START_EPOCH" \
    --run-tag "$RUN_TAG" \
    --num-workers 8 \
    --prefetch-factor 2 \
    2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ MSE 实验完成！"
    echo "日志文件: $LOG_FILE"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ MSE 实验失败！"
    echo "请查看日志文件: $LOG_FILE"
    echo "============================================================================"
    exit 1
fi


#!/bin/bash
# ============================================================================
# 掩码蒸馏验证脚本（小数据集 / 二进制数据格式）
# 使用转换后的二进制数据集（支持少量样本调试），10 epoch 后启动辨析训练
# 二进制数据集准备参考：docs/BINARY_DATASET_STRUCTURE.md
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
DATA_FORMAT="binary"
LOSS_TYPE="mse"  # 使用MSE作为基础损失
BACKBONE="yolov8"
CUDA_DEVICE=7  # 使用GPU 7进行测试

# 训练参数
EPOCHS=20  # 总共20个epoch，10个epoch后启动辨析训练
BATCH_SIZE=42
LEARNING_RATE=1e-3

# 损失权重
FEAT_WEIGHT=1.0
EDGE_WEIGHT=1.0

# 任务启动epoch
EDGE_TASK_START_EPOCH=5  # 第5个epoch启动边缘任务
MASK_TASK_START_EPOCH=10  # 第10个epoch启动掩码任务（辨析训练）

# 掩码任务配置
ENABLE_MASK_LOSS=true  # 启用掩码损失
MASK_LOSS_WEIGHT=1.0
MASK_BCE_WEIGHT=2.0
MASK_DICE_WEIGHT=1.0
MASK_HEAD_UNFREEZE_EPOCH=15  # 第15个epoch解冻掩码头
ENABLE_DETAILED_MASK_LOGGING=true  # 测试时启用详细诊断日志

# 二进制数据路径
# - SMALL：建议用于调试（可通过 convert_tar_to_bin.py --max-samples 生成）
# - FULL：完整数据集（同 run_fgd_experiment.sh）
USE_SMALL_BINARY_DATASET=true
BINARY_DATA_PATH_SMALL="/home/team/zouzhiyuan/dataset/sa1b_binary_test"
BINARY_DATA_PATH_FULL="/home/team/zouzhiyuan/dataset/sa1b_binary"

if [ "$USE_SMALL_BINARY_DATASET" = "true" ]; then
    DATASET_DESC="少量样本调试集"
    BINARY_DATA_PATH="$BINARY_DATA_PATH_SMALL"
    TOTAL_IMAGES=5000  # 小数据集估算值，用于进度条显示
    RUN_TAG="mask_task_binary_test_small"
else
    DATASET_DESC="完整二进制数据集"
    BINARY_DATA_PATH="$BINARY_DATA_PATH_FULL"
    TOTAL_IMAGES=109960  # 完整数据集图像数
    RUN_TAG="mask_task_binary_full"
fi

if [ ! -d "$BINARY_DATA_PATH" ]; then
    echo "错误: 二进制数据集目录不存在: $BINARY_DATA_PATH"
    echo "请确认是否已按照 docs/BINARY_DATASET_STRUCTURE.md 生成对应目录。"
    exit 1
fi

# ============================================================================
# 日志配置
# ============================================================================

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mask_task_test_${TIMESTAMP}.log"

# ============================================================================
# 启动训练
# ============================================================================

echo "============================================================================"
echo "辨析训练测试实验启动"
echo "============================================================================"
echo "GPU设备:        CUDA:${CUDA_DEVICE}"
echo "Loss类型:       ${LOSS_TYPE}"
echo "Backbone:       ${BACKBONE}"
echo "Epochs:         ${EPOCHS}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Learning Rate:  ${LEARNING_RATE}"
echo "总图像数:       ${TOTAL_IMAGES} (测试数据集估算值，用于进度条)"
echo "数据格式:       ${DATA_FORMAT}"
echo "数据目录:       ${BINARY_DATA_PATH}"
echo "数据集模式:     ${DATASET_DESC}"
echo "运行标签:       ${RUN_TAG}"
echo ""
echo "任务启动配置:"
echo "  边缘任务:     Epoch ${EDGE_TASK_START_EPOCH}"
echo "  掩码任务:     Epoch ${MASK_TASK_START_EPOCH} (辨析训练)"
echo ""
echo "掩码任务配置:"
echo "  启用掩码损失: ${ENABLE_MASK_LOSS}"
echo "  掩码损失权重: ${MASK_LOSS_WEIGHT}"
echo "  BCE权重:      ${MASK_BCE_WEIGHT}"
echo "  Dice权重:     ${MASK_DICE_WEIGHT}"
echo "  掩码头解冻:   Epoch ${MASK_HEAD_UNFREEZE_EPOCH}"
echo "  详细诊断日志: ${ENABLE_DETAILED_MASK_LOGGING}"
echo ""
echo "日志文件:       ${LOG_FILE}"
echo "============================================================================"
echo ""

# 构建训练命令
CMD_ARGS=(
    --env "$ENV_TYPE"
    --data-format "$DATA_FORMAT"
    --binary-data-path "$BINARY_DATA_PATH"
    --loss-type "$LOSS_TYPE"
    --backbone "$BACKBONE"
    --cuda-device "$CUDA_DEVICE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LEARNING_RATE"
    --feat-weight "$FEAT_WEIGHT"
    --edge-weight "$EDGE_WEIGHT"
    --total-images "$TOTAL_IMAGES"
    --edge-task-start-epoch "$EDGE_TASK_START_EPOCH"
    --mask-task-start-epoch "$MASK_TASK_START_EPOCH"
    --run-tag "$RUN_TAG"
    --mask-loss-weight "$MASK_LOSS_WEIGHT"
    --mask-bce-weight "$MASK_BCE_WEIGHT"
    --mask-dice-weight "$MASK_DICE_WEIGHT"
    --mask-head-unfreeze-epoch "$MASK_HEAD_UNFREEZE_EPOCH"
    
    # 【IO 优化参数】配合 RAM Cache 策略
    --num-workers 4
    --prefetch-factor 2
    
    --no-compile  # 测试脚本：禁用编译以快速启动
)

# 如果启用掩码损失，添加参数
if [ "$ENABLE_MASK_LOSS" = "true" ]; then
    CMD_ARGS+=(--enable-mask-loss)
fi

# 如果启用详细诊断日志，添加参数
if [ "$ENABLE_DETAILED_MASK_LOGGING" = "true" ]; then
    CMD_ARGS+=(--enable-detailed-mask-logging)
fi

# 执行训练命令
python "$TRAIN_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ 辨析训练测试实验完成！"
    echo "日志文件: $LOG_FILE"
    echo "============================================================================"
    echo ""
    echo "训练阶段总结:"
    echo "  Epoch 1-4:   仅特征对齐损失"
    echo "  Epoch 5-9:   特征对齐 + 边缘损失"
    echo "  Epoch 10-14:  特征对齐 + 边缘损失 + 掩码损失（掩码头冻结）"
    echo "  Epoch 15-20:  特征对齐 + 边缘损失 + 掩码损失（掩码头解冻）"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ 辨析训练测试实验失败！"
    echo "请查看日志文件: $LOG_FILE"
    echo "============================================================================"
    exit 1
fi


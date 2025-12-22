#!/bin/bash
# ============================================================================
# FGD 纯测试脚本
# 只测试 FGD 损失，不开启边缘任务和掩码任务
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
DATA_FORMAT="tar_shard"
LOSS_TYPE="fgd"
BACKBONE="yolov8"
CUDA_DEVICE=5  # 使用GPU 5进行测试

# 训练参数
EPOCHS=50  # 测试用，不需要500 epoch
BATCH_SIZE=42
LEARNING_RATE=1e-3
TOTAL_IMAGES=109960  # 数据集总图像数（用于进度条显示）

# 损失权重
FEAT_WEIGHT=1.0
EDGE_WEIGHT=0.0  # 关闭边缘任务（设为0）

# FGD 超参数（官方默认值）
FGD_ALPHA_FG=0.001      # 前景权重
FGD_BETA_BG=0.0005      # 背景权重（前景的一半）
FGD_ALPHA_EDGE=0.002    # 边缘权重（前景的两倍）
FGD_GAMMA_MASK=0.0
FGD_LAMBDA_RELA=0.0
FGD_TEMPERATURE=1.0

# 边缘增强（FGD特有）
ENABLE_EDGE_BOOST=true  # 启用边缘增强

# 任务启动epoch（设置为很大的值，确保不启动）
EDGE_TASK_START_EPOCH=999999  # 永不启动边缘任务
MASK_TASK_START_EPOCH=999999  # 永不启动掩码任务

# 数据路径（SSH环境默认路径）
FEATURES_DIR="/home/team/zouzhiyuan/dataset/sa1b_tar_shards"

# 输出目录标签（用于区分实验）
RUN_TAG="fgd_only_test"

# ============================================================================
# 日志配置
# ============================================================================

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fgd_only_test_${TIMESTAMP}.log"

# ============================================================================
# 启动训练
# ============================================================================

echo "============================================================================"
echo "FGD 纯测试实验启动（仅FGD损失，无边缘/掩码任务）"
echo "============================================================================"
echo "GPU设备:        CUDA:${CUDA_DEVICE}"
echo "Loss类型:       ${LOSS_TYPE}"
echo "Backbone:       ${BACKBONE}"
echo "Epochs:         ${EPOCHS}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Learning Rate:  ${LEARNING_RATE}"
echo "总图像数:       ${TOTAL_IMAGES}"
echo "数据格式:       ${DATA_FORMAT}"
echo "数据目录:       ${FEATURES_DIR}"
echo "运行标签:       ${RUN_TAG}"
echo "边缘增强:       ${ENABLE_EDGE_BOOST}"
echo "边缘任务:       ❌ 关闭 (edge_task_start_epoch=${EDGE_TASK_START_EPOCH})"
echo "掩码任务:       ❌ 关闭 (mask_task_start_epoch=${MASK_TASK_START_EPOCH})"
echo "FGD参数:"
echo "  Alpha FG:     ${FGD_ALPHA_FG}"
echo "  Beta BG:      ${FGD_BETA_BG}"
echo "  Alpha Edge:   ${FGD_ALPHA_EDGE}"
echo "  Temperature:  ${FGD_TEMPERATURE}"
echo "日志文件:       ${LOG_FILE}"
echo "============================================================================"
echo ""

# 构建训练命令
CMD_ARGS=(
    --env "$ENV_TYPE"
    --data-format "$DATA_FORMAT"
    --features-dir "$FEATURES_DIR"
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
    --fgd-alpha-fg "$FGD_ALPHA_FG"
    --fgd-beta-bg "$FGD_BETA_BG"
    --fgd-alpha-edge "$FGD_ALPHA_EDGE"
    --fgd-gamma-mask "$FGD_GAMMA_MASK"
    --fgd-lambda-rela "$FGD_LAMBDA_RELA"
    --fgd-temperature "$FGD_TEMPERATURE"
    
    # 【IO 优化参数】配合 RAM Cache 策略
    --num-workers 4
    --prefetch-factor 2
    
    --no-compile  # 测试脚本：禁用编译以快速启动
)

# 如果启用边缘增强，添加参数
if [ "$ENABLE_EDGE_BOOST" = "true" ]; then
    CMD_ARGS+=(--enable-edge-boost)
fi

# 执行训练命令
python "$TRAIN_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ FGD 纯测试实验完成！"
    echo "日志文件: $LOG_FILE"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ FGD 纯测试实验失败！"
    echo "请查看日志文件: $LOG_FILE"
    echo "============================================================================"
    exit 1
fi


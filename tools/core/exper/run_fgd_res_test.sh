#!/bin/bash
# ============================================================================
# FGD 蒸馏实验启动脚本（RT-DETR v2 Backbone）
# GPU: 1 (物理二号卡)
# Loss Type: FGD (启用边缘增强和掩码任务)
# 目标脚本: train_distill_res_test.py
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 解析命令行参数
# ============================================================================
RESUME_PATH=""
STRICT_RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume|-r)
            if [[ -z "${2:-}" ]]; then
                echo "错误: --resume 需要提供路径"
                exit 1
            fi
            RESUME_PATH="$2"
            shift 2
            ;;
        --strict-resume)
            STRICT_RESUME=true
            shift
            ;;
        --help|-h)
            cat <<'EOF'
用法: run_fgd_res_test.sh [--resume <路径>] [--strict-resume]
  --resume/-r         继续指定输出目录或 checkpoint 文件
  --strict-resume     传递 --strict-resume 给 train_distill_res_test.py
EOF
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -e "$RESUME_PATH" ]]; then
        echo "错误: --resume 路径不存在: $RESUME_PATH"
        exit 1
    fi
    if [[ -d "$RESUME_PATH" ]]; then
        if [[ "$(basename "$RESUME_PATH")" == "models" ]]; then
            RESUME_PARENT="$(cd "$RESUME_PATH/.." && pwd)"
            echo "[INFO] 检测到传入 models 子目录，自动改为其上级输出目录: $RESUME_PARENT"
            RESUME_PATH="$RESUME_PARENT"
        else
            RESUME_PATH="$(cd "$RESUME_PATH" && pwd)"
        fi
    else
        RESUME_PATH="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$RESUME_PATH")"
    fi
fi

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_distill_res_test.py"

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
LOSS_TYPE="fgd"
CUDA_DEVICE=2  # 物理二号卡（索引从0开始，1表示第二个GPU）

# RT-DETR v2 学生骨干配置
STUDENT_BACKBONE_TYPE="PResNet"  # 此次改用 PResNet（ResNet-vd）
STUDENT_VARIANT="d"              # PResNet 变体
STUDENT_DEPTH=50                 # ResNet50
FREEZE_NORM=false               # 如需冻结 BN，可设为 true
STUDENT_PRETRAINED="/home/team/zouzhiyuan/vfmkd/weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"
STUDENT_RESUME=""               # 蒸馏阶段断点，可留空

# 训练参数
EPOCHS=500
BATCH_SIZE=12  # 降低到12以避免OOM（边缘任务激活后需要额外显存）
LEARNING_RATE=1e-3
TOTAL_IMAGES=109960  # 数据集总图像数（用于进度条显示）

# 损失权重
FEAT_WEIGHT=1.0
EDGE_WEIGHT=1.0

# FGD 超参数（官方默认值）
FGD_ALPHA_FG=0.001      # 前景权重
FGD_BETA_BG=0.0005      # 背景权重（前景的一半）
FGD_ALPHA_EDGE=0.002    # 边缘权重（前景的两倍）
FGD_GAMMA_MASK=0.0
FGD_LAMBDA_RELA=0.0
FGD_TEMPERATURE=1.0

# 边缘增强（FGD特有）
ENABLE_EDGE_BOOST=true  # 启用边缘增强

# 任务启动epoch（特征训练1个周期后，即从第2个epoch开始启用）
EDGE_TASK_START_EPOCH=1  # 从第1个epoch开始（epoch 0之后）
MASK_TASK_START_EPOCH=1  # 从第1个epoch开始（经过1个周期的特征训练后）

# 掩码任务配置
ENABLE_MASK_LOSS=true  # 启用掩码辨析任务

# 可视化配置
VIS_INTERVAL=1  # 每1个epoch可视化一次（设置为0禁用可视化）

# 数据路径（二进制数据集路径）
BINARY_DATA_PATH="/home/team/zouzhiyuan/dataset/sa1b_binary"

# 输出目录标签（用于区分实验）
RUN_TAG="fgd_gpu1_resnet50_edge_boost_mask"

# ============================================================================
# 日志配置
# ============================================================================

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fgd_res_test_${TIMESTAMP}.log"

# ============================================================================
# 启动训练
# ============================================================================

echo "============================================================================"
echo "FGD 蒸馏实验启动 (train_distill_res_test.py)"
echo "============================================================================"
echo "GPU设备:        CUDA:${CUDA_DEVICE}"
echo "Loss类型:       ${LOSS_TYPE}"
echo "学生骨干:       ${STUDENT_BACKBONE_TYPE} (${STUDENT_VARIANT})"
if [ -n "$STUDENT_PRETRAINED" ]; then
    echo "预训练权重:     是 (${STUDENT_PRETRAINED})"
else
    echo "预训练权重:     否"
fi
echo "Epochs:         ${EPOCHS}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Learning Rate:  ${LEARNING_RATE}"
echo "总图像数:       ${TOTAL_IMAGES}"
echo "数据格式:       ${DATA_FORMAT}"
echo "数据目录:       ${BINARY_DATA_PATH}"
echo "运行标签:       ${RUN_TAG}"
echo "边缘增强:       ${ENABLE_EDGE_BOOST}"
echo "掩码任务:       ${ENABLE_MASK_LOSS}"
echo "可视化间隔:     每 ${VIS_INTERVAL} 个epoch"
echo "任务启动Epoch:"
echo "  边缘任务:     ${EDGE_TASK_START_EPOCH}"
echo "  掩码任务:     ${MASK_TASK_START_EPOCH}"
echo "FGD参数:"
echo "  Alpha FG:     ${FGD_ALPHA_FG}"
echo "  Beta BG:      ${FGD_BETA_BG}"
echo "  Alpha Edge:   ${FGD_ALPHA_EDGE}"
echo "  Temperature:  ${FGD_TEMPERATURE}"
echo "日志文件:       ${LOG_FILE}"
if [ -n "$RESUME_PATH" ]; then
    echo "恢复训练:       ${RESUME_PATH}"
    if [ "$STRICT_RESUME" = true ]; then
        echo "Resume模式:    strict"
    fi
fi
echo "============================================================================"
echo ""

# 构建训练命令
CMD_ARGS=(
    --env "$ENV_TYPE"
    --data-format "$DATA_FORMAT"
    --binary-data-path "$BINARY_DATA_PATH"
    --loss-type "$LOSS_TYPE"
    --backbone "rtdetrv2"
    --student-backbone-type "$STUDENT_BACKBONE_TYPE"
    --variant "$STUDENT_VARIANT"
    --depth "$STUDENT_DEPTH"
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
    --vis-interval "$VIS_INTERVAL"
    --fgd-alpha-fg "$FGD_ALPHA_FG"
    --fgd-beta-bg "$FGD_BETA_BG"
    --fgd-alpha-edge "$FGD_ALPHA_EDGE"
    --fgd-gamma-mask "$FGD_GAMMA_MASK"
    --fgd-lambda-rela "$FGD_LAMBDA_RELA"
    --fgd-temperature "$FGD_TEMPERATURE"
    --num-workers 8
    --prefetch-factor 2
    --no-compile
)

# 可选：冻结 BN
if [ "$FREEZE_NORM" = true ]; then
    CMD_ARGS+=(--freeze-norm)
fi

# 可选：学生 ImageNet 预训练
if [ -n "$STUDENT_PRETRAINED" ]; then
    CMD_ARGS+=(--student-pretrained "$STUDENT_PRETRAINED")
fi

# 可选：蒸馏断点恢复
if [ -n "$STUDENT_RESUME" ]; then
    CMD_ARGS+=(--student-resume "$STUDENT_RESUME")
fi

# 如果启用边缘增强，添加参数
if [ "$ENABLE_EDGE_BOOST" = "true" ]; then
    CMD_ARGS+=(--enable-edge-boost)
fi

# 如果启用掩码任务，添加参数
if [ "$ENABLE_MASK_LOSS" = "true" ]; then
    CMD_ARGS+=(--enable-mask-loss)
fi

# 训练断点恢复
if [ -n "$RESUME_PATH" ]; then
    CMD_ARGS+=(--resume "$RESUME_PATH")
fi
if [ "$STRICT_RESUME" = true ]; then
    CMD_ARGS+=(--strict-resume)
fi

# 执行训练命令
python "$TRAIN_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ FGD (RT-DETR v2) 实验完成！"
    echo "日志文件: $LOG_FILE"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ FGD (RT-DETR v2) 实验失败！"
    echo "请查看日志文件: $LOG_FILE"
    echo "============================================================================"
    exit 1
fi



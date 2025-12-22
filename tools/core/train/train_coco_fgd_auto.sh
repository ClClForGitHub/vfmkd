#!/bin/bash
# ============================================================================
# COCO 微调对比实验脚本
#   - 实验A: 加载蒸馏得到的backbone
#   - 实验B: 随机初始化backbone
#   - 两个实验除“是否加载预训练权重”与GPU设备外，其余超参完全一致
# ============================================================================
set -euo pipefail

# ------------------------------ 基础配置 ------------------------------
# 修复 GLIBCXX 版本问题：确保脚本里直接设置 LD_LIBRARY_PATH
CONDA_LIB_PATH="/home/team/zouzhiyuan/anaconda3/envs/s2detkd/lib"
export LD_LIBRARY_PATH="${CONDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
TRAIN_SCRIPT="tools/core/train/train_coco_mmdet_lego.py"
DEFAULT_PRETRAINED="/home/team/zouzhiyuan/vfmkd/outputs/distill_single_test_FGD/20251124_121436_yolov8_edge_boost_fgd_gpu4_edge_boost/models/best_backbone_mmdet.pth"
WORK_DIR_BASE=${WORK_DIR_BASE:-"./work_dirs/coco_finetune_compare"}

# 可配置超参（通过环境变量覆盖）
PRETRAINED_BACKBONE=${PRETRAINED_BACKBONE:-"${DEFAULT_PRETRAINED}"}
BATCH_SIZE=${BATCH_SIZE:-32}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}
UNFREEZE_EPOCH=${UNFREEZE_EPOCH:-1}
RUN_MODE=${RUN_MODE:-"both"}          # both | pretrained | random
PRETRAINED_GPUS=${PRETRAINED_GPUS:-"5"}
RANDOM_GPUS=${RANDOM_GPUS:-"6"}
# 若希望两个实验使用相同GPU，可将 PRETRAINED_GPUS 与 RANDOM_GPUS 设为同一个值

# 统一的训练参数（除GPU与是否加载权重外，其余完全相同）
COMMON_ARGS=("--bs" "${BATCH_SIZE}")
if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
    COMMON_ARGS+=("--freeze-backbone" "--unfreeze-at-epoch" "${UNFREEZE_EPOCH}")
fi

print_summary() {
    echo "============================================================================"
    echo "COCO 微调对比实验"
    echo "----------------------------------------------------------------------------"
    echo "工作目录基准 : ${WORK_DIR_BASE}"
    echo "Batch Size   : ${BATCH_SIZE}"
    echo "冻结Backbone : ${FREEZE_BACKBONE}"
    if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
        echo "解冻Epoch    : ${UNFREEZE_EPOCH}"
    fi
    echo "运行模式     : ${RUN_MODE} (both | pretrained | random)"
    echo "蒸馏模型路径 : ${PRETRAINED_BACKBONE}"
    echo "预训练GPU    : ${PRETRAINED_GPUS}"
    echo "随机GPU      : ${RANDOM_GPUS}"
    echo "============================================================================"
    echo
}

validate_backbone_path() {
    if [[ ! -f "${PRETRAINED_BACKBONE}" ]]; then
        echo "❌ 错误: 找不到预训练模型 ${PRETRAINED_BACKBONE}"
        exit 1
    fi
}

run_pretrained() {
    validate_backbone_path
    local work_dir="${WORK_DIR_BASE}_pretrained"
    echo "[预训练实验] 工作目录: ${work_dir}"
    echo "[预训练实验] 使用 GPU: ${PRETRAINED_GPUS}"
    echo "[预训练实验] 启动中..."

    local cmd=(python "${TRAIN_SCRIPT}" \
        "--distilled-backbone" "${PRETRAINED_BACKBONE}" \
        "--work-dir" "${work_dir}" \
        "${COMMON_ARGS[@]}")

    CUDA_VISIBLE_DEVICES="${PRETRAINED_GPUS}" "${cmd[@]}"
    echo "✅ 预训练实验完成: ${work_dir}"
    echo
}

run_random() {
    local work_dir="${WORK_DIR_BASE}_random"
    echo "[随机初始化实验] 工作目录: ${work_dir}"
    echo "[随机初始化实验] 使用 GPU: ${RANDOM_GPUS}"
    echo "[随机初始化实验] 启动中..."

    local cmd=(python "${TRAIN_SCRIPT}" \
        "--work-dir" "${work_dir}" \
        "--random-init" \
        "${COMMON_ARGS[@]}")

    CUDA_VISIBLE_DEVICES="${RANDOM_GPUS}" "${cmd[@]}"
    echo "✅ 随机初始化实验完成: ${work_dir}"
    echo
}

print_summary

case "${RUN_MODE}" in
    both)
        run_pretrained
        run_random
        ;;
    pretrained)
        run_pretrained
        ;;
    random)
        run_random
        ;;
    *)
        echo "❌ RUN_MODE 取值无效: ${RUN_MODE} (允许 both | pretrained | random)"
        exit 1
        ;;
esac

echo "============================================================================"
echo "🎉 实验结束"
echo "预训练结果: ${WORK_DIR_BASE}_pretrained"
echo "随机初始化: ${WORK_DIR_BASE}_random"
echo "============================================================================"
echo

#!/bin/bash
# 简单的训练脚本包装器，自动保存日志

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 从配置文件读取 output_dir（如果第一个参数是 -c）
OUTPUT_DIR=""
if [[ "$1" == "-c" && -n "${2:-}" ]]; then
    CONFIG_FILE="$2"
    if [[ -f "${CONFIG_FILE}" ]]; then
        OUTPUT_DIR=$(grep -E "^output_dir:" "${CONFIG_FILE}" | head -1 | sed 's/^output_dir:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        # 如果是相对路径，转换为绝对路径
        if [[ -n "${OUTPUT_DIR}" && ! "${OUTPUT_DIR}" =~ ^/ ]]; then
            OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
        fi
    fi
fi

# 如果仍然没有 OUTPUT_DIR，使用默认值
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/output/train_$(date +%Y%m%d_%H%M%S)"
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

echo "============================================================================"
echo "训练日志将保存到: ${LOG_FILE}"
echo "============================================================================"

# 执行训练命令，并重定向到日志文件
"$@" 2>&1 | tee "${LOG_FILE}"


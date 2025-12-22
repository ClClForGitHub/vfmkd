#!/bin/bash
# 批量转换 TAR → BIN 脚本

set -e  # 遇到错误立即退出

# 配置
TAR_DIR="/home/team/zouzhiyuan/dataset/sa1b_tar_shards"
OUTPUT_DIR="/home/team/zouzhiyuan/dataset/sa1b_binary"
MODEL_TYPE="sam2.1_hiera_b+"
WORKERS=32  # 并行 Worker 进程数
SCRIPT_PATH="tools/core/data/convert_tar_to_bin.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "批量转换 TAR → BIN"
echo "=========================================="
echo "TAR 目录: ${TAR_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Worker 进程数: ${WORKERS}"
echo "=========================================="
echo ""

# 检查 TAR 目录
if [ ! -d "${TAR_DIR}" ]; then
    echo -e "${RED}错误: TAR 目录不存在: ${TAR_DIR}${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 获取所有 TAR 文件
TAR_FILES=($(ls ${TAR_DIR}/sa1b_shard_*.tar* 2>/dev/null | sort))
TOTAL_SHARDS=${#TAR_FILES[@]}

if [ ${TOTAL_SHARDS} -eq 0 ]; then
    echo -e "${RED}错误: 在 ${TAR_DIR} 中未找到 TAR 文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 ${TOTAL_SHARDS} 个 TAR 文件${NC}"
echo ""

# 检查脚本是否存在
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo -e "${RED}错误: 转换脚本不存在: ${SCRIPT_PATH}${NC}"
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_SHARDS=()

# 转换每个 shard
for i in "${!TAR_FILES[@]}"; do
    TAR_FILE="${TAR_FILES[$i]}"
    SHARD_NUM=$(printf "%05d" $i)
    
    echo -e "${YELLOW}[$((i+1))/${TOTAL_SHARDS}] 处理 shard_${SHARD_NUM}...${NC}"
    echo "  TAR 文件: $(basename ${TAR_FILE})"
    
    # 构造参数数组
    CMD_ARGS=(
        "--tar-path" "${TAR_FILE}"
        "--output-dir" "${OUTPUT_DIR}"
        "--model-type" "${MODEL_TYPE}"
        "--workers" "${WORKERS}"
        "--quiet"
    )
    
    # 从第二个 shard 开始（i > 0），添加 --append 参数
    if [ $i -gt 0 ]; then
        CMD_ARGS+=("--append")
        echo "  模式: 追加（追加到已有数据）"
    else
        echo "  模式: 覆盖（创建新文件）"
    fi
    
    # 运行转换
    if python "${SCRIPT_PATH}" "${CMD_ARGS[@]}"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "  ${GREEN}✓ 成功${NC}"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_SHARDS+=("shard_${SHARD_NUM}")
        echo -e "  ${RED}✗ 失败${NC}"
    fi
    
    echo ""
done

# 计算总时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# 输出统计信息
echo "=========================================="
echo "转换完成！"
echo "=========================================="
echo "总 Shard 数: ${TOTAL_SHARDS}"
echo -e "成功: ${GREEN}${SUCCESS_COUNT}${NC}"
echo -e "失败: ${RED}${FAIL_COUNT}${NC}"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo -e "${RED}失败的 Shard:${NC}"
    for shard in "${FAILED_SHARDS[@]}"; do
        echo "  - ${shard}"
    done
    echo ""
fi

echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 检查输出文件
if [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo ""
    echo "输出文件检查:"
    ls -lh "${OUTPUT_DIR}"/*.bin "${OUTPUT_DIR}/config.json" "${OUTPUT_DIR}/keys.txt" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    
    # 统计总大小
    TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
    echo ""
    echo "总输出大小: ${TOTAL_SIZE}"
fi


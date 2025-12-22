#!/bin/bash
# 监控 TAR → BIN 转换进度

OUTPUT_DIR="/home/team/zouzhiyuan/dataset/sa1b_binary"
LOG_FILE="/tmp/tar_to_bin_conversion.log"

echo "=========================================="
echo "转换进度监控"
echo "=========================================="
echo ""

# 检查进程
echo "运行中的进程:"
ps aux | grep -E "convert_tar_to_bin|batch_convert" | grep -v grep | wc -l | xargs echo "  Worker 进程数:"
echo ""

# 检查输出文件
if [ -d "${OUTPUT_DIR}" ]; then
    echo "输出文件状态:"
    if [ -f "${OUTPUT_DIR}/config.json" ]; then
        echo "  ✓ config.json 存在"
    else
        echo "  ✗ config.json 不存在"
    fi
    
    if [ -f "${OUTPUT_DIR}/keys.txt" ]; then
        KEY_COUNT=$(wc -l < "${OUTPUT_DIR}/keys.txt")
        echo "  ✓ keys.txt 存在 (${KEY_COUNT} 行)"
    else
        echo "  ✗ keys.txt 不存在"
    fi
    
    echo ""
    echo "二进制文件大小:"
    for file in "${OUTPUT_DIR}"/*.bin; do
        if [ -f "$file" ]; then
            SIZE=$(ls -lh "$file" | awk '{print $5}')
            echo "  $(basename $file): ${SIZE}"
        fi
    done
    
    echo ""
    TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)
    echo "总输出大小: ${TOTAL_SIZE}"
else
    echo "输出目录不存在: ${OUTPUT_DIR}"
fi

echo ""
echo "=========================================="
echo "最近日志 (最后 20 行):"
echo "=========================================="
if [ -f "${LOG_FILE}" ]; then
    tail -20 "${LOG_FILE}"
else
    echo "日志文件不存在: ${LOG_FILE}"
fi


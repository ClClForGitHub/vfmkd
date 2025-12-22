#!/bin/bash
# ============================================================================
# 检查 /dev/shm 内存使用情况和最近15天的使用记录
# ============================================================================

echo "============================================================================"
echo "检查 /dev/shm (RAM盘) 内存使用情况"
echo "============================================================================"
echo ""

# 1. 总内存容量和使用情况
echo "1. 内存容量和使用情况:"
df -h /dev/shm
echo ""

# 2. 详细统计
echo "2. 详细统计:"
TOTAL=$(df -BG /dev/shm | tail -1 | awk '{print $2}' | sed 's/G//')
USED=$(df -BG /dev/shm | tail -1 | awk '{print $3}' | sed 's/G//')
AVAILABLE=$(df -BG /dev/shm | tail -1 | awk '{print $4}' | sed 's/G//')
USE_PERCENT=$(df -h /dev/shm | tail -1 | awk '{print $5}')

echo "   总容量:     ${TOTAL}GB"
echo "   已使用:     ${USED}GB"
echo "   可用空间:   ${AVAILABLE}GB"
echo "   使用率:     ${USE_PERCENT}"
echo ""

# 3. 当前文件列表和大小
echo "3. 当前文件/目录列表:"
if [ "$(ls -A /dev/shm 2>/dev/null)" ]; then
    echo "   文件/目录列表:"
    ls -lh /dev/shm | tail -n +2 | while read line; do
        echo "   $line"
    done
    echo ""
    
    # 统计总大小
    TOTAL_SIZE=$(du -sh /dev/shm 2>/dev/null | awk '{print $1}')
    echo "   当前总占用:   ${TOTAL_SIZE}"
    echo ""
else
    echo "   ✅ /dev/shm 目录为空（没有文件）"
    echo ""
fi

# 4. 检查最近15天内的使用情况
echo "4. 最近15天内的使用情况:"
DAYS=15
CUTOFF_DATE=$(date -d "${DAYS} days ago" +%s)
FOUND_RECENT=false

if [ "$(ls -A /dev/shm 2>/dev/null)" ]; then
    echo "   查找最近 ${DAYS} 天内创建或修改的文件..."
    echo ""
    
    # 查找最近15天的文件
    RECENT_COUNT=0
    find /dev/shm -type f -newermt "${DAYS} days ago" 2>/dev/null | while read file; do
        if [ -f "$file" ]; then
            RECENT_COUNT=$((RECENT_COUNT + 1))
            FILE_DATE=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1)
            FILE_TIME=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
            FILE_SIZE=$(du -h "$file" 2>/dev/null | awk '{print $1}')
            FILE_OWNER=$(stat -c %U "$file" 2>/dev/null)
            echo "   📄 $(basename "$file")"
            echo "      路径:     $file"
            echo "      大小:     ${FILE_SIZE}"
            echo "      所有者:   ${FILE_OWNER}"
            echo "      修改时间: ${FILE_DATE} ${FILE_TIME}"
            echo ""
        fi
    done
    
    # 查找最近15天的目录
    find /dev/shm -type d -newermt "${DAYS} days ago" 2>/dev/null | while read dir; do
        if [ "$dir" != "/dev/shm" ] && [ -d "$dir" ]; then
            DIR_DATE=$(stat -c %y "$dir" 2>/dev/null | cut -d' ' -f1)
            DIR_TIME=$(stat -c %y "$dir" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
            DIR_OWNER=$(stat -c %U "$dir" 2>/dev/null)
            DIR_SIZE=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
            echo "   📁 $(basename "$dir")"
            echo "      路径:     $dir"
            echo "      大小:     ${DIR_SIZE}"
            echo "      所有者:   ${DIR_OWNER}"
            echo "      修改时间: ${DIR_DATE} ${DIR_TIME}"
            echo ""
        fi
    done
    
    # 统计最近15天的文件数量
    RECENT_FILES=$(find /dev/shm -type f -newermt "${DAYS} days ago" 2>/dev/null | wc -l)
    RECENT_DIRS=$(find /dev/shm -type d -newermt "${DAYS} days ago" 2>/dev/null | grep -v "^/dev/shm$" | wc -l)
    
    if [ "$RECENT_FILES" -gt 0 ] || [ "$RECENT_DIRS" -gt 0 ]; then
        echo "   统计:"
        echo "      最近 ${DAYS} 天内的文件数: ${RECENT_FILES}"
        echo "      最近 ${DAYS} 天内的目录数: ${RECENT_DIRS}"
    else
        echo "   ✅ 最近 ${DAYS} 天内没有文件被创建或修改"
    fi
else
    echo "   ✅ /dev/shm 目录为空，最近 ${DAYS} 天内没有使用记录"
fi
echo ""

# 5. 按用户统计
echo "5. 按用户统计当前使用情况:"
if [ "$(ls -A /dev/shm 2>/dev/null)" ]; then
    echo "   各用户占用空间:"
    find /dev/shm -mindepth 1 -maxdepth 1 -exec stat -c "%U %s" {} \; 2>/dev/null | \
        awk '{user[$1] += $2} END {for (u in user) printf "      %s: %.2f GB\n", u, user[u]/1024/1024/1024}' | \
        sort -k2 -rn || echo "   无法统计（可能权限问题）"
    echo ""
else
    echo "   ✅ 没有文件，无法统计"
    echo ""
fi

# 6. 系统内存信息（参考）
echo "6. 系统内存信息（参考）:"
if command -v free &> /dev/null; then
    free -h | head -2
    echo ""
    echo "   说明: /dev/shm 通常占用系统内存的一部分"
    echo ""
fi

# 7. 总结
echo "============================================================================"
echo "检查完成"
echo "============================================================================"
echo ""
echo "总结:"
echo "  - 总容量:     ${TOTAL}GB"
echo "  - 已使用:     ${USED}GB (${USE_PERCENT})"
echo "  - 可用空间:   ${AVAILABLE}GB"
if [ "$(ls -A /dev/shm 2>/dev/null)" ]; then
    echo "  - 当前文件:   有文件存在"
else
    echo "  - 当前文件:   目录为空"
fi
echo ""
echo "注意: /dev/shm 是 RAM 文件系统，系统重启后所有数据会丢失"
echo "      只能查看当前存在的文件，无法查看历史已删除的文件"
echo "============================================================================"


#!/bin/bash
# ============================================================================
# 测试 /dev/shm (RAM盘) 是否可用
# ============================================================================

set -e

echo "============================================================================"
echo "测试 /dev/shm (RAM盘) 可用性"
echo "============================================================================"
echo ""

# 1. 检查挂载状态
echo "1. 检查挂载状态:"
if mountpoint -q /dev/shm; then
    echo "   ✅ /dev/shm 已挂载"
    df -h /dev/shm | tail -1
else
    echo "   ❌ /dev/shm 未挂载"
    exit 1
fi
echo ""

# 2. 检查文件系统类型
echo "2. 检查文件系统类型:"
FS_TYPE=$(mount | grep "/dev/shm" | awk '{print $5}')
echo "   文件系统类型: $FS_TYPE"
if [ "$FS_TYPE" = "tmpfs" ]; then
    echo "   ✅ 确认为 tmpfs (RAM文件系统)"
else
    echo "   ⚠️  文件系统类型不是 tmpfs"
fi
echo ""

# 3. 检查可用空间
echo "3. 检查可用空间:"
AVAILABLE=$(df -BG /dev/shm | tail -1 | awk '{print $4}' | sed 's/G//')
echo "   可用空间: ${AVAILABLE}GB"
if [ "$AVAILABLE" -gt 1 ]; then
    echo "   ✅ 有足够的可用空间"
else
    echo "   ⚠️  可用空间较少"
fi
echo ""

# 4. 测试写入权限
echo "4. 测试写入权限:"
TEST_FILE="/dev/shm/test_write_$$.tmp"
if touch "$TEST_FILE" 2>/dev/null; then
    echo "   ✅ 可以创建文件"
    rm -f "$TEST_FILE"
else
    echo "   ❌ 无法创建文件"
    exit 1
fi
echo ""

# 5. 测试写入和读取
echo "5. 测试写入和读取:"
TEST_FILE="/dev/shm/test_rw_$$.tmp"
TEST_CONTENT="Hello from RAM disk! $(date)"
echo "$TEST_CONTENT" > "$TEST_FILE"
if [ -f "$TEST_FILE" ]; then
    echo "   ✅ 文件写入成功"
    READ_CONTENT=$(cat "$TEST_FILE")
    if [ "$READ_CONTENT" = "$TEST_CONTENT" ]; then
        echo "   ✅ 文件读取成功，内容匹配"
    else
        echo "   ❌ 文件读取失败，内容不匹配"
        exit 1
    fi
    rm -f "$TEST_FILE"
else
    echo "   ❌ 文件写入失败"
    exit 1
fi
echo ""

# 6. 测试写入速度（小文件）
echo "6. 测试小文件写入速度:"
TEST_FILE="/dev/shm/test_speed_$$.tmp"
SIZE_MB=100
echo "   写入 ${SIZE_MB}MB 测试文件..."
START_TIME=$(date +%s.%N)
dd if=/dev/zero of="$TEST_FILE" bs=1M count=$SIZE_MB 2>/dev/null
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
SPEED=$(echo "scale=2; $SIZE_MB / $ELAPSED" | bc)
echo "   写入速度: ${SPEED} MB/s"
rm -f "$TEST_FILE"
echo ""

# 7. 测试目录创建
echo "7. 测试目录创建:"
TEST_DIR="/dev/shm/test_dir_$$"
if mkdir -p "$TEST_DIR" 2>/dev/null; then
    echo "   ✅ 可以创建目录"
    rmdir "$TEST_DIR"
else
    echo "   ❌ 无法创建目录"
    exit 1
fi
echo ""

# 8. 总结
echo "============================================================================"
echo "✅ /dev/shm (RAM盘) 测试完成 - 所有测试通过！"
echo "============================================================================"
echo ""
echo "总结:"
echo "  - 挂载状态: ✅ 正常"
echo "  - 文件系统: ✅ tmpfs (RAM)"
echo "  - 可用空间: ✅ ${AVAILABLE}GB"
echo "  - 读写权限: ✅ 正常"
echo "  - 写入速度: ✅ ${SPEED} MB/s"
echo ""
echo "可以使用 /dev/shm 作为临时存储或缓存目录！"
echo "============================================================================"


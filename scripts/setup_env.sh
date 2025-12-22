#!/bin/bash
# VFMKD环境设置脚本

echo "设置VFMKD环境..."

# 创建必要的目录
mkdir -p datasets/coco
mkdir -p datasets/sa1b
mkdir -p weights
mkdir -p teacher_features/sam
mkdir -p teacher_features/dino
mkdir -p experiments/logs

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "环境设置完成！"
echo "请运行以下命令安装依赖："
echo "pip install -r requirements.txt"


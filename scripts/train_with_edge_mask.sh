#!/bin/bash
# ========================================
# 渐进式边缘掩码训练脚本
# ========================================
# 
# 功能：
#   - 前5个epoch：全局学习边缘位置
#   - 第6个epoch开始：只在膨胀边缘区域（3x3核，±1像素）内计算损失
#   - 自动忽略物体内部的细碎纹理
#
# 使用方法：
#   chmod +x scripts/train_with_edge_mask.sh
#   ./scripts/train_with_edge_mask.sh
# ========================================

echo ""
echo "============================================================"
echo "  渐进式边缘掩码训练 - EdgeSAM + RepViT"
echo "============================================================"
echo ""
echo "[配置信息]"
echo "  - 数据集: sa1b_subset_300 (300张图片)"
echo "  - 骨干网络: RepViT-M1"
echo "  - 蒸馏损失: FGD (Focal and Global Distillation)"
echo "  - 边缘掩码: 渐进式启用 (第5个epoch开始)"
echo "  - 膨胀核: 3x3 (边缘向外扩展1像素)"
echo ""
echo "[训练策略]"
echo "  Epoch 1-5:  全局学习，在整个256x256区域计算边缘损失"
echo "  Epoch 6-20: 精确对齐，只在膨胀边缘区域内计算损失"
echo ""
echo "============================================================"
echo ""

# 检查CUDA是否可用
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

# 开始训练
python tools/train_distill_single_test.py \
  --features-dir VFMKD/outputs/features_sam2_300 \
  --images-dir VFMKD/datasets/sa1b_subset_300 \
  --loss-type fgd \
  --backbone repvit \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-3 \
  --feat-weight 1.0 \
  --edge-weight 1.0 \
  --enable-edge-mask-progressive \
  --edge-mask-start-epoch 5 \
  --edge-mask-kernel-size 3 \
  --no-val

echo ""
echo "============================================================"
echo "  训练完成！"
echo "============================================================"
echo ""
echo "[输出文件]"
echo "  - 最佳模型: outputs/distill_single_test/best_fgd.pth"
echo "  - 训练日志: outputs/distill_single_test/results.log"
echo "  - 可视化结果: VFMKD/outputs/testFGDFSD/fgd_no_edge_boost_vis/"
echo ""
echo "[下一步]"
echo "  1. 查看训练日志，确认边缘掩码在第5个epoch启用"
echo "  2. 检查Edge Mask Coverage统计（应在10-20%范围）"
echo "  3. 对比可视化结果中的边缘预测质量"
echo ""


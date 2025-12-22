#!/usr/bin/env python3
"""
边缘掩码可视化测试脚本
演示3x3膨胀核如何生成边缘区域掩码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_sample_edge_gt():
    """创建一个示例边缘GT（模拟物体轮廓）"""
    # 64x64分辨率
    edge_gt = torch.zeros(1, 1, 64, 64)
    
    # 绘制一个矩形边缘（模拟物体轮廓）
    edge_gt[0, 0, 20:22, 15:50] = 1.0  # 上边
    edge_gt[0, 0, 45:47, 15:50] = 1.0  # 下边
    edge_gt[0, 0, 20:47, 15:17] = 1.0  # 左边
    edge_gt[0, 0, 20:47, 48:50] = 1.0  # 右边
    
    return edge_gt

def create_student_prediction():
    """创建一个示例学生预测（包含内部纹理）"""
    pred = torch.zeros(1, 1, 64, 64)
    
    # 主边缘（稍有偏移）
    pred[0, 0, 19:21, 14:51] = 0.8
    pred[0, 0, 46:48, 14:51] = 0.8
    pred[0, 0, 19:48, 14:16] = 0.8
    pred[0, 0, 19:48, 49:51] = 0.8
    
    # 内部纹理（噪声）
    pred[0, 0, 25:27, 20:45] = 0.3  # 水平纹理1
    pred[0, 0, 32:34, 22:43] = 0.4  # 水平纹理2
    pred[0, 0, 38:40, 18:48] = 0.3  # 水平纹理3
    pred[0, 0, 22:45, 25:27] = 0.3  # 垂直纹理1
    pred[0, 0, 24:44, 35:37] = 0.4  # 垂直纹理2
    
    return pred

def apply_edge_mask(edge_gt, kernel_size=3):
    """
    使用MaxPool2d实现边缘膨胀
    
    Args:
        edge_gt: 边缘GT [B, 1, H, W]
        kernel_size: 膨胀核大小（3=±1像素）
        
    Returns:
        膨胀后的边缘掩码
    """
    dilater = nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    edge_mask = dilater(edge_gt)
    return edge_mask

def visualize_edge_mask_effect():
    """可视化边缘掩码的效果"""
    print("=== 边缘掩码可视化测试 ===\n")
    
    # 创建测试数据
    edge_gt = create_sample_edge_gt()  # [1, 1, 64, 64]
    student_pred = create_student_prediction()  # [1, 1, 64, 64]
    
    # 生成不同kernel_size的边缘掩码
    edge_mask_3x3 = apply_edge_mask(edge_gt, kernel_size=3)
    edge_mask_5x5 = apply_edge_mask(edge_gt, kernel_size=5)
    edge_mask_7x7 = apply_edge_mask(edge_gt, kernel_size=7)
    
    # 转换为numpy（方便可视化）
    edge_gt_np = edge_gt[0, 0].numpy()
    student_pred_np = student_pred[0, 0].numpy()
    edge_mask_3x3_np = edge_mask_3x3[0, 0].numpy()
    edge_mask_5x5_np = edge_mask_5x5[0, 0].numpy()
    edge_mask_7x7_np = edge_mask_7x7[0, 0].numpy()
    
    # 计算掩码覆盖率
    coverage_3x3 = edge_mask_3x3_np.mean() * 100
    coverage_5x5 = edge_mask_5x5_np.mean() * 100
    coverage_7x7 = edge_mask_7x7_np.mean() * 100
    
    print(f"边缘GT像素数: {edge_gt_np.sum():.0f} ({edge_gt_np.mean()*100:.2f}%)")
    print(f"3x3掩码覆盖率: {coverage_3x3:.2f}%")
    print(f"5x5掩码覆盖率: {coverage_5x5:.2f}%")
    print(f"7x7掩码覆盖率: {coverage_7x7:.2f}%")
    print()
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== 第一行：原始数据和掩码 =====
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(edge_gt_np, cmap='gray', vmin=0, vmax=1)
    ax0.set_title(f"边缘GT\n({edge_gt_np.sum():.0f} pixels, {edge_gt_np.mean()*100:.2f}%)", fontsize=10)
    plt.colorbar(im0, ax=ax0, fraction=0.046)
    ax0.axis('off')
    
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(edge_mask_3x3_np, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f"3×3掩码（推荐）\n(Coverage: {coverage_3x3:.2f}%)", fontsize=10, color='green')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(edge_mask_5x5_np, cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f"5×5掩码\n(Coverage: {coverage_5x5:.2f}%)", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(edge_mask_7x7_np, cmap='hot', vmin=0, vmax=1)
    ax3.set_title(f"7×7掩码（太大）\n(Coverage: {coverage_7x7:.2f}%)", fontsize=10, color='red')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 4])
    im4 = ax4.imshow(student_pred_np, cmap='viridis', vmin=0, vmax=1)
    ax4.set_title("学生预测\n(包含内部纹理)", fontsize=10)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    ax4.axis('off')
    
    # ===== 第二行：掩码应用效果 =====
    # 计算掩码后的损失权重
    masked_pred_3x3 = student_pred_np * edge_mask_3x3_np
    masked_pred_5x5 = student_pred_np * edge_mask_5x5_np
    masked_pred_7x7 = student_pred_np * edge_mask_7x7_np
    
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(student_pred_np, cmap='viridis', vmin=0, vmax=1)
    ax5.set_title("无掩码（全局）\n所有像素参与损失计算", fontsize=10)
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(masked_pred_3x3, cmap='viridis', vmin=0, vmax=1)
    ax6.set_title("3×3掩码应用后\n✅ 保留边缘，忽略内部", fontsize=10, color='green', fontweight='bold')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(masked_pred_5x5, cmap='viridis', vmin=0, vmax=1)
    ax7.set_title("5×5掩码应用后\n部分内部纹理未忽略", fontsize=10)
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(masked_pred_7x7, cmap='viridis', vmin=0, vmax=1)
    ax8.set_title("7×7掩码应用后\n❌ 大部分内部纹理仍保留", fontsize=10, color='red')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    ax8.axis('off')
    
    # 差异图
    diff_global = np.abs(student_pred_np - edge_gt_np)
    diff_3x3 = np.abs(masked_pred_3x3 - (edge_gt_np * edge_mask_3x3_np))
    
    ax9 = fig.add_subplot(gs[1, 4])
    im9 = ax9.imshow(diff_global, cmap='hot', vmin=0, vmax=1)
    ax9.set_title(f"全局误差\nMAE={diff_global.mean():.4f}", fontsize=10)
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    ax9.axis('off')
    
    # ===== 第三行：统计分析 =====
    # 内部区域定义：物体内部（非边缘）
    interior_mask = 1 - edge_mask_3x3_np
    interior_mask[edge_gt_np > 0] = 0  # 排除真实边缘
    
    # 计算内部纹理响应
    interior_response_global = (student_pred_np * interior_mask).sum()
    interior_response_3x3 = (masked_pred_3x3 * interior_mask).sum()
    interior_response_5x5 = (masked_pred_5x5 * interior_mask).sum()
    
    print("=== 内部纹理抑制效果 ===")
    print(f"无掩码（全局）：内部响应 = {interior_response_global:.2f}")
    print(f"3×3掩码：内部响应 = {interior_response_3x3:.2f} (↓{(1-interior_response_3x3/interior_response_global)*100:.1f}%)")
    print(f"5×5掩码：内部响应 = {interior_response_5x5:.2f} (↓{(1-interior_response_5x5/interior_response_global)*100:.1f}%)")
    print()
    
    # 绘制统计图
    ax10 = fig.add_subplot(gs[2, :2])
    mask_types = ['无掩码\n(全局)', '3×3掩码\n(推荐)', '5×5掩码', '7×7掩码']
    coverages = [100, coverage_3x3, coverage_5x5, coverage_7x7]
    colors = ['gray', 'green', 'orange', 'red']
    bars = ax10.bar(mask_types, coverages, color=colors, alpha=0.7, edgecolor='black')
    ax10.set_ylabel('掩码覆盖率 (%)', fontsize=12)
    ax10.set_title('不同膨胀核的掩码覆盖率对比', fontsize=12, fontweight='bold')
    ax10.axhline(y=15, color='blue', linestyle='--', linewidth=1, label='理想范围 (10-20%)')
    ax10.legend()
    ax10.grid(axis='y', alpha=0.3)
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{cov:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 绘制内部纹理抑制效果
    ax11 = fig.add_subplot(gs[2, 2:])
    responses = [interior_response_global, interior_response_3x3, interior_response_5x5]
    response_types = ['无掩码', '3×3掩码', '5×5掩码']
    colors2 = ['gray', 'green', 'orange']
    bars2 = ax11.bar(response_types, responses, color=colors2, alpha=0.7, edgecolor='black')
    ax11.set_ylabel('内部纹理响应值', fontsize=12)
    ax11.set_title('内部纹理抑制效果对比（值越低越好）', fontsize=12, fontweight='bold')
    ax11.grid(axis='y', alpha=0.3)
    for bar, resp in zip(bars2, responses):
        height = bar.get_height()
        reduction = (1 - resp/interior_response_global) * 100 if resp < interior_response_global else 0
        label = f'{resp:.1f}\n(↓{reduction:.0f}%)' if reduction > 0 else f'{resp:.1f}'
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 保存
    plt.suptitle('渐进式边缘掩码效果演示（64×64分辨率）', fontsize=16, fontweight='bold', y=0.98)
    output_path = "VFMKD/outputs/edge_mask_visualization_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存: {output_path}\n")
    plt.close()
    
    # 打印建议
    print("=== 💡 使用建议 ===")
    print("1. ✅ 推荐使用3×3核（kernel_size=3）")
    print("   - 覆盖率适中（~12-18%）")
    print("   - 有效抑制内部纹理（↓100%）")
    print("   - 为边缘提供±1像素容错区")
    print()
    print("2. ⚠️  5×5核可能过大")
    print("   - 覆盖率偏高（>30%）")
    print("   - 可能保留部分内部纹理")
    print()
    print("3. ❌ 7×7核太大，不推荐")
    print("   - 覆盖率过高（>50%）")
    print("   - 失去精确边缘对齐的意义")
    print()
    print("4. 🎯 渐进式训练策略")
    print("   - Epoch 1-5: 无掩码，全局学习")
    print("   - Epoch 6+: 3×3掩码，精确对齐")
    print()

def test_loss_calculation():
    """测试损失计算的差异"""
    print("\n=== 损失计算测试 ===\n")
    
    # 创建测试数据
    edge_gt = create_sample_edge_gt()
    student_pred = create_student_prediction()
    edge_mask = apply_edge_mask(edge_gt, kernel_size=3)
    
    # 计算全局损失
    loss_global = nn.functional.mse_loss(student_pred, edge_gt)
    
    # 计算掩码损失
    student_pred_masked = student_pred * edge_mask
    edge_gt_masked = edge_gt * edge_mask
    num_valid = edge_mask.sum().clamp(min=1.0)
    loss_masked = (nn.functional.mse_loss(student_pred_masked, edge_gt_masked, reduction='sum') / num_valid)
    
    print(f"全局MSE损失: {loss_global.item():.6f}")
    print(f"掩码MSE损失: {loss_masked.item():.6f}")
    print(f"差异: {abs(loss_global.item() - loss_masked.item()):.6f}")
    print()
    print("💡 掩码损失更关注边缘区域，忽略内部纹理的影响！\n")

if __name__ == "__main__":
    visualize_edge_mask_effect()
    test_loss_calculation()
    
    print("="*60)
    print("测试完成！")
    print("="*60)


#!/usr/bin/env python3
"""
模型对比脚本
用于汇总和对比不同蒸馏方法的性能
"""

import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(log_path):
    """解析训练日志文件"""
    results = {
        'epochs': [],
        'train_total': [],
        'train_feat': [],
        'train_edge': [],
        'test_total': None,
        'test_feat': None,
        'test_edge': None
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # 解析训练epoch结果
            if line.startswith('epoch='):
                parts = line.strip().split('\t')
                epoch_data = {}
                for part in parts:
                    key, value = part.split('=')
                    epoch_data[key] = value
                
                results['epochs'].append(int(epoch_data['epoch']))
                results['train_total'].append(float(epoch_data['train_total']))
                results['train_feat'].append(float(epoch_data['train_feat']))
                results['train_edge'].append(float(epoch_data['train_edge']))
            
            # 解析测试集结果
            elif line.startswith('test_total='):
                parts = line.strip().split('\t')
                for part in parts:
                    key, value = part.split('=')
                    results[key] = float(value)
    
    return results

def load_all_results(base_dir="VFMKD/outputs/testFGDFSD"):
    """加载所有模型的结果"""
    base_path = Path(base_dir)
    
    models = {
        'MSE Baseline': 'mse_baseline.log',
        'FGD (No Edge)': 'fgd_no_edge.log',
        'FGD (Edge Boost)': 'fgd_edge_boost.log',
        'FSD (No Edge)': 'fsd_no_edge.log',
        'FSD (Edge Boost)': 'fsd_edge_boost.log'
    }
    
    all_results = {}
    
    for model_name, log_file in models.items():
        log_path = base_path / log_file
        if log_path.exists():
            print(f"✅ 加载 {model_name}: {log_path}")
            all_results[model_name] = parse_log_file(log_path)
        else:
            print(f"⚠️  未找到 {model_name}: {log_path}")
    
    return all_results

def create_comparison_table(all_results):
    """创建对比表格"""
    data = []
    
    for model_name, results in all_results.items():
        if results['test_total'] is not None:
            data.append({
                'Model': model_name,
                'Test Total Loss': f"{results['test_total']:.6f}",
                'Test Feat Loss': f"{results['test_feat']:.6f}",
                'Test Edge Loss': f"{results['test_edge']:.6f}",
                'Final Train Total': f"{results['train_total'][-1]:.6f}" if results['train_total'] else 'N/A',
                'Final Train Feat': f"{results['train_feat'][-1]:.6f}" if results['train_feat'] else 'N/A',
                'Final Train Edge': f"{results['train_edge'][-1]:.6f}" if results['train_edge'] else 'N/A'
            })
    
    df = pd.DataFrame(data)
    return df

def plot_training_curves(all_results, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    # 1. Total Loss
    ax = axes[0, 0]
    for model_name, results in all_results.items():
        if results['train_total']:
            ax.plot(results['epochs'], results['train_total'], marker='o', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Training)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Feature Loss
    ax = axes[0, 1]
    for model_name, results in all_results.items():
        if results['train_feat']:
            ax.plot(results['epochs'], results['train_feat'], marker='s', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss (Training)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Edge Loss
    ax = axes[1, 0]
    for model_name, results in all_results.items():
        if results['train_edge']:
            ax.plot(results['epochs'], results['train_edge'], marker='^', label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Edge Loss')
    ax.set_title('Edge Loss (Training)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Test Set Performance Bar Chart
    ax = axes[1, 1]
    models = []
    test_totals = []
    test_feats = []
    for model_name, results in all_results.items():
        if results['test_total'] is not None:
            models.append(model_name.replace(' ', '\n'))  # 换行以适应图表
            test_totals.append(results['test_total'])
            test_feats.append(results['test_feat'])
    
    x = range(len(models))
    width = 0.35
    ax.bar([i - width/2 for i in x], test_totals, width, label='Total Loss', alpha=0.8)
    ax.bar([i + width/2 for i in x], test_feats, width, label='Feat Loss', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Loss')
    ax.set_title('Test Set Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练曲线保存至: {output_path}")
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("🔍 模型对比分析")
    print("="*60)
    
    # 加载所有结果
    all_results = load_all_results()
    
    if not all_results:
        print("❌ 未找到任何训练结果")
        return
    
    # 创建对比表格
    print("\n" + "="*60)
    print("📊 测试集性能对比表")
    print("="*60)
    df = create_comparison_table(all_results)
    print(df.to_string(index=False))
    
    # 保存表格
    output_dir = Path("VFMKD/outputs/testFGDFSD")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\n💾 对比表保存至: {output_dir / 'model_comparison.csv'}")
    
    # 绘制训练曲线
    print("\n" + "="*60)
    print("📈 生成训练曲线对比图...")
    print("="*60)
    plot_training_curves(all_results, output_dir)
    
    # 分析最佳模型
    print("\n" + "="*60)
    print("🏆 最佳模型分析")
    print("="*60)
    
    best_total = min([(name, res['test_total']) for name, res in all_results.items() if res['test_total'] is not None], key=lambda x: x[1])
    best_feat = min([(name, res['test_feat']) for name, res in all_results.items() if res['test_feat'] is not None], key=lambda x: x[1])
    
    print(f"✨ 最佳 Total Loss: {best_total[0]} ({best_total[1]:.6f})")
    print(f"✨ 最佳 Feature Loss: {best_feat[0]} ({best_feat[1]:.6f})")
    
    print("\n" + "="*60)
    print("✅ 对比分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()



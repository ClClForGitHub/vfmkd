#!/usr/bin/env python3
"""
正类重加权对比实验脚本
运行4组实验：
1. Baseline: MSE + 无边带 + 无正类重加权
2. Pos-Weight: MSE + 无边带 + 正类重加权
3. Progressive: MSE + 渐进式边带（Epoch 5启用）+ 无正类重加权
4. Progressive + Pos-Weight: MSE + 渐进式边带（Epoch 5启用）+ 正类重加权
"""

import subprocess
import sys
from pathlib import Path

# 实验配置
EXPERIMENTS = [
    {
        "name": "Baseline",
        "args": [
            "--loss-type", "mse",
            "--epochs", "5",
            "--batch-size", "4",
            "--lr", "1e-3",
            "--no-val",
        ],
        "log": "VFMKD/outputs/pos_weight_comparison/baseline.log",
    },
    {
        "name": "Pos-Weight",
        "args": [
            "--loss-type", "mse",
            "--epochs", "5",
            "--batch-size", "4",
            "--lr", "1e-3",
            "--no-val",
            "--use-pos-weight",
        ],
        "log": "VFMKD/outputs/pos_weight_comparison/pos_weight.log",
    },
    {
        "name": "Progressive",
        "args": [
            "--loss-type", "mse",
            "--epochs", "8",  # 需要8 epochs才能看到渐进效果
            "--batch-size", "4",
            "--lr", "1e-3",
            "--no-val",
            "--enable-edge-mask-progressive",
            "--edge-mask-start-epoch", "5",
            "--edge-mask-kernel-size", "3",
        ],
        "log": "VFMKD/outputs/pos_weight_comparison/progressive.log",
    },
    {
        "name": "Progressive+Pos-Weight",
        "args": [
            "--loss-type", "mse",
            "--epochs", "8",
            "--batch-size", "4",
            "--lr", "1e-3",
            "--no-val",
            "--enable-edge-mask-progressive",
            "--edge-mask-start-epoch", "5",
            "--edge-mask-kernel-size", "3",
            "--use-pos-weight",
        ],
        "log": "VFMKD/outputs/pos_weight_comparison/progressive_pos_weight.log",
    },
]

# 公共参数
COMMON_ARGS = [
    "--features-dir", "VFMKD/outputs/features_v1_300",
    "--images-dir", r"datasets\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0",
]

def run_experiment(exp_config):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_config['name']}")
    print(f"{'='*80}\n")
    
    # 创建日志目录
    log_path = Path(exp_config['log'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable,
        "VFMKD/tools/core/train_distill_single_test.py",
    ] + COMMON_ARGS + exp_config['args']
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # 运行实验（前台运行，输出到控制台和日志文件）
    with open(log_path, 'w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        
        process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] Experiment '{exp_config['name']}' failed with return code {process.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] Experiment '{exp_config['name']}' completed successfully!")
        return True

def extract_final_metrics(log_path):
    """从日志文件提取最终测试指标"""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    
    metrics = {}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # 提取 Test Results
            if "Test Results:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("total="):
                        metrics['test_total'] = float(part.split('=')[1])
                    elif part.startswith("feat="):
                        metrics['test_feat'] = float(part.split('=')[1])
                    elif part.startswith("edge="):
                        metrics['test_edge'] = float(part.split('=')[1])
            
            # 提取 Unified Metrics
            if "Feature MSE:" in line:
                metrics['feat_mse'] = float(line.split(':')[1].strip())
            elif "Feature MAE:" in line:
                metrics['feat_mae'] = float(line.split(':')[1].strip())
            elif "Cosine Similarity:" in line:
                metrics['cosine_sim'] = float(line.split(':')[1].strip())
            elif "Edge Loss:" in line:
                metrics['edge_loss'] = float(line.split(':')[1].strip())
    
    return metrics

def print_comparison_table():
    """打印对比表格"""
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    results = []
    for exp in EXPERIMENTS:
        metrics = extract_final_metrics(exp['log'])
        if metrics:
            results.append({
                'name': exp['name'],
                **metrics
            })
    
    if not results:
        print("No results found!")
        return
    
    # 打印表格
    print(f"{'Model':<25} | {'Test Total':>10} | {'Feat Loss':>10} | {'Edge Loss':>10} | {'Feat MSE':>10} | {'Cosine Sim':>11}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['name']:<25} | "
              f"{r.get('test_total', 0):.4f}     | "
              f"{r.get('test_feat', 0):.4f}     | "
              f"{r.get('test_edge', 0):.4f}     | "
              f"{r.get('feat_mse', 0):.6f}   | "
              f"{r.get('cosine_sim', 0):.6f}")
    
    print("\n" + "="*80)
    
    # 分析结果
    print("\nKEY FINDINGS:\n")
    
    # 找到最佳模型
    best_edge = min(results, key=lambda x: x.get('edge_loss', float('inf')))
    best_feat = min(results, key=lambda x: x.get('feat_mse', float('inf')))
    
    print(f"1. Best Edge Loss: {best_edge['name']} ({best_edge.get('edge_loss', 0):.4f})")
    print(f"2. Best Feature MSE: {best_feat['name']} ({best_feat.get('feat_mse', 0):.6f})")
    
    # 对比Baseline vs Pos-Weight
    baseline = next((r for r in results if r['name'] == 'Baseline'), None)
    pos_weight = next((r for r in results if r['name'] == 'Pos-Weight'), None)
    
    if baseline and pos_weight:
        edge_improve = (baseline.get('edge_loss', 0) - pos_weight.get('edge_loss', 0)) / baseline.get('edge_loss', 1) * 100
        print(f"\n3. Pos-Weight vs Baseline:")
        print(f"   - Edge Loss Improvement: {edge_improve:+.1f}%")
    
    # 对比Progressive vs Progressive+Pos-Weight
    prog = next((r for r in results if r['name'] == 'Progressive'), None)
    prog_pos = next((r for r in results if r['name'] == 'Progressive+Pos-Weight'), None)
    
    if prog and prog_pos:
        edge_improve = (prog.get('edge_loss', 0) - prog_pos.get('edge_loss', 0)) / prog.get('edge_loss', 1) * 100
        print(f"\n4. Progressive+Pos-Weight vs Progressive:")
        print(f"   - Edge Loss Improvement: {edge_improve:+.1f}%")
    
    print("\n" + "="*80 + "\n")

def main():
    """主函数"""
    print("="*80)
    print("Positive Class Reweighting Comparison Experiment")
    print("="*80)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print("Experiments:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}")
    print()
    
    # 运行所有实验
    success_count = 0
    for exp in EXPERIMENTS:
        if run_experiment(exp):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Completed {success_count}/{len(EXPERIMENTS)} experiments successfully")
    print(f"{'='*80}\n")
    
    # 打印对比结果
    if success_count > 0:
        print_comparison_table()

if __name__ == "__main__":
    main()


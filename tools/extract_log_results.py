#!/usr/bin/env python3
"""
从训练日志中提取测试结果并生成对比表格
"""

import re
from pathlib import Path
from typing import Dict, Optional


def extract_test_results(log_file: Path) -> Optional[Dict[str, float]]:
    """从日志文件中提取Test Results"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找Test Results行
    pattern = r'Test Results: total=([\d.]+) feat=([\d.]+) edge=([\d.]+)'
    match = re.search(pattern, content)
    
    if match:
        return {
            'total': float(match.group(1)),
            'feat': float(match.group(2)),
            'edge': float(match.group(3))
        }
    return None


def main():
    log_dir = Path("VFMKD/outputs/testFGDFSD")
    
    # 定义要提取的日志文件
    logs = {
        "MSE": log_dir / "mse_5ep.log",
        "FGD": log_dir / "fgd_5ep.log",
        "FGD+Edge": log_dir / "fgd_with_edge_5ep.log"
    }
    
    # 提取结果
    results = {}
    for name, log_file in logs.items():
        result = extract_test_results(log_file)
        if result:
            results[name] = result
            print(f"✓ Extracted {name}: {result}")
        else:
            print(f"✗ Failed to extract {name} from {log_file}")
    
    if not results:
        print("\n[ERROR] No results extracted!")
        return
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("TEST RESULTS COMPARISON (on 30 test images)")
    print(f"{'='*80}\n")
    
    print(f"{'Method':<15} {'Total Loss':<15} {'Feature Loss':<15} {'Edge Loss':<15}")
    print(f"{'-'*80}")
    
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['total']:<15.4f} {metrics['feat']:<15.4f} {metrics['edge']:<15.4f}")
    
    # 计算相对改进
    if "MSE" in results:
        print(f"\n{'Improvement vs MSE':<15}")
        print(f"{'-'*80}")
        mse_feat = results["MSE"]["feat"]
        mse_edge = results["MSE"]["edge"]
        
        for method, metrics in results.items():
            if method == "MSE":
                continue
            feat_improve = (mse_feat - metrics['feat']) / mse_feat * 100
            edge_improve = (mse_edge - metrics['edge']) / mse_edge * 100
            print(f"{method:<15} Feat: {feat_improve:+6.2f}%   Edge: {edge_improve:+6.2f}%")
    
    print(f"\n{'='*80}")
    
    # 找出最优方法
    print("\nBest Performance:")
    print(f"  Feature Loss: {min(results.items(), key=lambda x: x[1]['feat'])[0]} ({min(r['feat'] for r in results.values()):.4f})")
    print(f"  Edge Loss:    {min(results.items(), key=lambda x: x[1]['edge'])[0]} ({min(r['edge'] for r in results.values()):.4f})")
    print(f"  Total Loss:   {min(results.items(), key=lambda x: x[1]['total'])[0]} ({min(r['total'] for r in results.values()):.4f})")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()


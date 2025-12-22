#!/usr/bin/env python3
"""
下载SAM权重文件
"""

import os
import sys
import requests
from pathlib import Path
import argparse


def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """下载文件"""
    print(f"正在下载: {url}")
    print(f"保存到: {filepath}")
    
    # 创建目录
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
    
    print(f"\n[SUCCESS] 下载完成: {filepath}")
    print(f"文件大小: {filepath.stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser("下载SAM权重文件")
    parser.add_argument("--output-dir", type=str, default="EdgeSAM-master/weights", 
                       help="输出目录")
    parser.add_argument("--model", type=str, default="vit_h", 
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM模型类型")
    
    args = parser.parse_args()
    
    # SAM权重下载链接
    sam_weights = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    output_dir = Path(args.output_dir)
    model_type = args.model
    
    if model_type not in sam_weights:
        print(f"[ERROR] 不支持的模型类型: {model_type}")
        print(f"支持的模型: {list(sam_weights.keys())}")
        return
    
    url = sam_weights[model_type]
    filename = url.split('/')[-1]
    filepath = output_dir / filename
    
    # 检查文件是否已存在
    if filepath.exists():
        print(f"[INFO] 权重文件已存在: {filepath}")
        print(f"文件大小: {filepath.stat().st_size / (1024**2):.1f} MB")
        return
    
    try:
        download_file(url, filepath)
        print(f"[SUCCESS] SAM {model_type} 权重下载完成")
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Download pretrained models for VFMKD.
"""

import os
import urllib.request
from pathlib import Path
import argparse
from typing import Dict, List


# Model URLs
MODEL_URLS = {
    'sam_vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'sam_vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'sam_vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'dinov2_vitg14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14_pretrain.pth',
    'dinov2_vitl14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14_pretrain.pth',
    'dinov2_vitb14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14_pretrain.pth',
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
}


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        filepath: Local path to save file
        chunk_size: Chunk size for downloading
    """
    print(f"Downloading {url} to {filepath}")
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    try:
        urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        print(f"\nDownloaded successfully: {filepath}")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial file
        raise


def download_models(models: List[str], weights_dir: str = "weights") -> None:
    """
    Download specified models.
    
    Args:
        models: List of model names to download
        weights_dir: Directory to save models
    """
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        if model not in MODEL_URLS:
            print(f"Unknown model: {model}")
            print(f"Available models: {list(MODEL_URLS.keys())}")
            continue
        
        url = MODEL_URLS[model]
        filepath = weights_path / f"{model}.pt"
        
        # Skip if already exists
        if filepath.exists():
            print(f"Model {model} already exists: {filepath}")
            continue
        
        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"Failed to download {model}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download pretrained models for VFMKD")
    parser.add_argument("--models", nargs="+", default=list(MODEL_URLS.keys()),
                       help="Models to download")
    parser.add_argument("--weights-dir", default="weights",
                       help="Directory to save models")
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for model, url in MODEL_URLS.items():
            print(f"  {model}: {url}")
        return
    
    print(f"Downloading models: {args.models}")
    download_models(args.models, args.weights_dir)
    print("Download completed!")


if __name__ == "__main__":
    main()
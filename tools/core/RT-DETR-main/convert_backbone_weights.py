#!/usr/bin/env python3
"""
RT-DETRv2 Backbone权重转换脚本

将蒸馏得到的backbone权重转换为RT-DETRv2可以加载的格式。

使用方法:
    python convert_backbone_weights.py \
        --input /path/to/best_backbone_mmdet.pth \
        --output /path/to/rtdetrv2_backbone_weights.pth
"""

import argparse
import torch
from pathlib import Path


def _get_state_dict_from_checkpoint(checkpoint: dict, subkey: str | None):
    """
    提取真正的 state_dict：
      - 如果提供 subkey，则优先使用 checkpoint[subkey]
      - 若 subkey 不存在或未提供，尝试常见字段（state_dict/model/ema）
      - 否则假设整个 checkpoint 就是 state_dict
    """
    if subkey:
        nested = checkpoint
        for key in subkey.split('.'):
            if isinstance(nested, dict) and key in nested:
                nested = nested[key]
            else:
                raise KeyError(f"子键 '{subkey}' 不存在于 checkpoint 中")
        checkpoint = nested

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'ema' in checkpoint:
            ema_state = checkpoint['ema']
            if isinstance(ema_state, dict) and 'module' in ema_state:
                return ema_state['module']
            return ema_state
    if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("无法解析 checkpoint 中的 state_dict，请检查输入文件或使用 --subkey 指定子键。")


def convert_backbone_weights(input_path: str, output_path: str, prefix: str = "backbone.", subkey: str | None = None):
    """
    将backbone权重转换为RT-DETRv2格式
    
    Args:
        input_path: 输入的backbone权重文件路径
        output_path: 输出的权重文件路径
        prefix: 键名前缀，默认为"backbone."
    """
    print(f"加载权重文件: {input_path}")
    
    # 加载原始权重
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = _get_state_dict_from_checkpoint(checkpoint, subkey)
    
    print(f"原始权重键数量: {len(state_dict)}")
    print(f"前10个键名示例:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # 转换键名：添加backbone前缀
    converted_state_dict = {}
    for key, value in state_dict.items():
        # 如果键名已经包含backbone前缀，跳过
        if key.startswith(prefix):
            new_key = key
        else:
            new_key = prefix + key
        
        converted_state_dict[new_key] = value
    
    print(f"\n转换后权重键数量: {len(converted_state_dict)}")
    print(f"前10个转换后的键名示例:")
    for i, key in enumerate(list(converted_state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # 构建RT-DETRv2兼容的checkpoint格式
    # RT-DETRv2的load_tuning_state会检查'model'或'ema.module'键
    output_checkpoint = {
        'model': converted_state_dict,
    }
    
    # 保存转换后的权重
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存转换后的权重到: {output_path}")
    torch.save(output_checkpoint, output_path)
    print("✅ 权重转换完成！")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="将backbone权重转换为RT-DETRv2格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换backbone权重
  python convert_backbone_weights.py \\
      --input /path/to/best_backbone_mmdet.pth \\
      --output /path/to/rtdetrv2_backbone_weights.pth
  
  # 使用转换后的权重训练RT-DETRv2
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
      tools/train.py \\
      -c configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml \\
      -t /path/to/rtdetrv2_backbone_weights.pth \\
      --use-amp --seed=0
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的backbone权重文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出的权重文件路径'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='backbone.',
        help='键名前缀，默认为"backbone."'
    )
    parser.add_argument(
        '--subkey',
        type=str,
        default=None,
        help='可选：checkpoint 中包含真正 state_dict 的子键，例如 "backbone"'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 执行转换
    convert_backbone_weights(args.input, args.output, args.prefix, args.subkey)


if __name__ == '__main__':
    main()


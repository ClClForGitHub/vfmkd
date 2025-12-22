"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items()
                        if k not in ['update', ] and v is not None})

    # 如果从命令行传入了顶层 use_dog，则自动映射到 RTDETR.use_dog
    # 优先级：显式的 RTDETR.use_dog（来自 YAML 或 -u） > 顶层 --use-dog
    if 'use_dog' in update_dict:
        # 如果 RTDETR 还没在 update_dict 里，先创建一个空 dict
        if 'RTDETR' not in update_dict or not isinstance(update_dict['RTDETR'], dict):
            update_dict['RTDETR'] = {}
        # 只有当 RTDETR 内部还没有 use_dog 时，才用顶层 use_dog 填充
        if 'use_dog' not in update_dict['RTDETR']:
            update_dict['RTDETR']['use_dog'] = update_dict['use_dog']

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # Bio-Retina / DoG 开关（默认关闭）
    parser.add_argument('--use-dog', action='store_true',
                        help='enable BioRetina DoG enhancement in RT-DETR (default: off)')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

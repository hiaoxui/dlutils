from argparse import ArgumentParser, BooleanOptionalAction
import os

from .process_args import Default


def add_gpu_arguments(p: ArgumentParser, default_strategy: str):
    p.add_argument(
        '--strategy', type=str, default=Default(default_strategy), choices=['deepspeed', 'ddp'],
        help='multigpu backend: deepspeed or ddp'
    )
    p.add_argument(
        '--precision', type=str, default=Default('bf16-mixed'), help='default to bf16-mixed'
    )
    p.add_argument(
        '--n-gpu', type=int, default=Default(16),
        help='if input is more than existing gpus, will fall back to maximum available'
    )
    p.add_argument('--deepspeed', type=int, default=Default(2), help='deepspeed stage, default to zero 2')
    p.add_argument('--offload-optim', action=BooleanOptionalAction, default=Default(False), help='for deepspeed')
    p.add_argument('--offload-param', action=BooleanOptionalAction, default=Default(False), help='for deepspeed')


def add_common_args(p: ArgumentParser, project_name: str):
    p.add_argument('--ckpt', metavar='CKPT_PATH', type=str, help='checkpoint path')
    p.add_argument(
        '--cache', type=str, default=Default(os.path.join(os.environ.get('HOME', ''), project_name)),
        help='root cache path'
    )
    p.add_argument('--exp', type=str, default=Default('debug'), help='experiment name')
    p.add_argument('-o', type=str, default='/tmp', help='output path for prediction')
    p.add_argument('--debug', action='store_true', help='if on, will disable multiprocess dataloader')
    p.add_argument('--test', action=BooleanOptionalAction, default=Default(False), help='run on the test set')
    p.add_argument('--resume', action=BooleanOptionalAction, default=Default(True), help='resume optim states')
    p.add_argument('--warmup', type=int, default=Default(1000), help='#warmup steps')
    p.add_argument('--logger', type=str, default=Default('tensorboard'), choices=['tensorboard', 'wandb'])
    # other necessary args
    # val-interval n-val bsz eff-bsz

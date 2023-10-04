from argparse import ArgumentParser, BooleanOptionalAction
import os

from .progress_args import Default


def add_gpu_arguments(p: ArgumentParser, default_strategy: str):
    p.add_argument('--strategy', type=str, default=Default(default_strategy))
    p.add_argument('--precision', type=str, default=Default('bf16-mixed'))
    p.add_argument('--n-gpu', type=int, default=Default(16))
    p.add_argument('--deepspeed', type=int, default=Default(2))
    p.add_argument('--offload-optim', action=BooleanOptionalAction, default=Default(False))
    p.add_argument('--offload-param', action=BooleanOptionalAction, default=Default(False))


def add_common_args(p: ArgumentParser, project_name: str):
    p.add_argument('--ckpt', type=str)
    p.add_argument(
        '--cache', type=str,
        default=Default(os.path.join(os.environ.get('HOME', ''), project_name))
    )
    p.add_argument('--exp', type=str, default=Default('debug'))
    p.add_argument('-o', type=str, default='/tmp')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--test', type=BooleanOptionalAction, default=Default(False))
    # other necessary args
    # val-interval n-val bsz eff-bsz

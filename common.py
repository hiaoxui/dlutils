import logging
import os
import warnings
import re
from collections import defaultdict

import torch


def num_seq(nums):
    seq = []
    last_n = -99999
    for n in sorted(nums):
        if n == last_n + 1:
            seq[-1][1] = n
        else:
            seq.append([n, None])
        last_n = n
    return '[' + ','.join([str(s) if e is None else f'{s}-{e}' for s, e in seq]) + ']'


def param_names(params, trainable=True):
    if isinstance(params, torch.nn.Module):
        params = [(name, pa) for name, pa in params.named_parameters() if pa.requires_grad or not trainable]
    seen, ret, nums = set(), [], defaultdict(list)
    for name, pa in params:
        if re.findall(r'\d+', name):
            n = int(re.findall(r'\d+', name)[0])
            name = re.sub(r'(\d+)', '#', name, 1)
            nums[name].append(n)
        if name not in seen:
            seen.add(name)
            ret.append([name, tuple(pa.shape)])
    return [(name if name not in nums else name.replace('#', num_seq(nums[name])), sha) for name, sha in ret]


def configure_logger():
    logger_ = logging.getLogger('pl')
    stm_hdl = logging.StreamHandler()
    logger_.addHandler(stm_hdl)
    logger_.setLevel('INFO')
    return logger_


def get_process_logger():
    logger_ = logging.getLogger('process')
    stm_hdl = logging.StreamHandler()
    local_rank = os.environ.get('LOCAL_RANK')
    fmt = logging.Formatter(f'[rk={local_rank}] %(message)s')
    stm_hdl.setFormatter(fmt)
    logger_.addHandler(stm_hdl)
    if 'DEBUG_PROCESS' in os.environ:
        logger_.setLevel('DEBUG')
    else:
        logger_.setLevel('WARNING')
    return logger_


logger = configure_logger()
process_logger = get_process_logger()

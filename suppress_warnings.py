import warnings
import logging


filtered_out = [
    'The `srun` command is available on your system but is not used',
    'dataloader, does not have many workers',
    'YPU available', 'TPU available', 'IPU available', 'HPU available',
    'exists and is not empty', 'Setting ds_accelerator', 'that has Tensor Cores',
    '- CUDA_VISIBLE_DEVICES', 'Found keys that are in the model state'
]

for fo_ in filtered_out:
    warnings.filterwarnings('ignore', f'.*{fo_}.*')


class SupressFilter(logging.Filter):
    def filter(self, record):
        for fo in filtered_out:
            if fo.lower() in record.msg.lower():
                return False
        return True


for logger_name in [
    'DeepSpeed', 'lightning_utilities.core.rank_zero', 'lightning.pytorch.utilities.rank_zero',
    'lightning.pytorch.accelerators.cuda'
]:
    logging.getLogger(logger_name).addFilter(SupressFilter())


def suppress():
    try:
        import datasets
        datasets.logging.set_verbosity_error()
    except ImportError:
        pass
    try:
        import deepspeed
        logging.getLogger('DeepSpeed').setLevel('WARNING')
    except ImportError:
        pass

import warnings
import logging


warnings.filterwarnings('ignore', r'.*Setting ds_accelerator.*')
warnings.filterwarnings('ignore', r'.*The `srun` command is available on your system but is not used.*')
warnings.filterwarnings('ignore', r'.*The dataloader, (train|val)_dataloader, does not have many workers.*')
warnings.filterwarnings('ignore', r'.*[YIH]PU available*')
warnings.filterwarnings('ignore', r'.*TPU available*')
warnings.filterwarnings('ignore', r'.*Checkpoint directory .* exists and is not empty')


class SupressFilter(logging.Filter):
    def filter(self, record):
        if 'Setting ds_accelerator' in record.msg:
            return False
        return True


logging.getLogger('DeepSpeed').addFilter(SupressFilter())


def suppress():
    try:
        import datasets
        datasets.logging.set_verbosity_error()
    except ImportError:
        pass
    try:
        from deepspeed.utils import logger as ds_logger
        ds_logger.setLevel('WARNING')
    except ImportError:
        pass

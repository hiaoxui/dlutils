from typing import *
from dataclasses import dataclass

from lightning.pytorch import loggers as pl_loggers

from dlutils.cli.ckpt import process_ckpt
from dlutils.cli.gpu import gen_gpu_args


@dataclass()
class Default:
    value: Any = None

    def __repr__(self):
        return str(self.value)


def process_args(args):
    hparams, version = dict(), None
    extras = dict()
    if args.ckpt is not None:
        hparams, log_dir, version, ckpt = process_ckpt(args)
        if not args.resume:
            version = None
        extras['model_path'], extras['resume_from'] = ckpt.model_path, ckpt.top
        if args.o is None:
            extras['predict_path'] = ckpt.predict_path
        if not args.resume:
            extras.pop('resume_from')
    if args.o is not None:
        extras['predict_path'] = args.o
    if args.test:
        extras['predict_path'] += '.test'

    # run over all examples fore predict
    if args.action == 'predict' and isinstance(args.n_val, Default):
        args.n_val = 9999999999999
    # if args.perplexity and 'predict_path' in extras:
    #     extras['predict_path'] = extras['predict_path'].replace('predict', 'perplexity')

    for k in set(vars(args)) | set(hparams):
        if not hasattr(args, k):
            setattr(args, k, hparams.get(k))
            continue
        v = getattr(args, k, None)
        if isinstance(v, Default) and k in hparams:
            setattr(args, k, hparams.get(k))
        elif isinstance(v, Default):
            setattr(args, k, getattr(args, k).value)

    extras['gpu_config'], extras['gpu_kwargs'], extras['n_actual_gpu'] = gen_gpu_args(
        args.n_gpu, args.precision, args.strategy, args.deepspeed, args.offload_optim, args.offload_param
    )
    if args.action == 'train':
        tensorboard = pl_loggers.TensorBoardLogger(
            args.cache if not args.debug else '/tmp/debug', name=args.exp, version=version
        )
        extras['default_root_dir'] = tensorboard.log_dir
    else:
        tensorboard = None
        extras['default_root_dir'] = '/tmp/nugget'
    if args.ckpt is None and tensorboard is not None:
        tensorboard.log_hyperparams(args)

    extras['accumulate'] = max(1, args.eff_bsz // (max(extras['n_actual_gpu'], 1) * args.bsz))

    return tensorboard, extras

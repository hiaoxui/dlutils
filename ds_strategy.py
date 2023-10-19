import time
from typing import *
import logging

import torch
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy, MisconfigurationException
from lightning.fabric.utilities.distributed import (
    _distributed_available,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)


class MyDeepSpeedStrategy(DeepSpeedStrategy):

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        debug = kwargs.get('debug', False)
        if debug:
            logging.warning('before available')
        if not _distributed_available():
            return
        if debug:
            logging.warning('after available')
        if torch.distributed.get_backend() == "nccl":
            if debug:
                logging.warning('doing nccl')
            import ilock
            with ilock.ILock('barrier', lock_directory='/home/gqin2'):
                did = self.determine_ddp_device_ids()
                time.sleep(1.0)
            if debug:
                logging.warning(f'got did {did}')
            work = torch.distributed.barrier(device_ids=did, async_op=True)
            if debug:
                logging.warning(f'got work {work}')
            work.wait()
        else:
            torch.distributed.barrier()

    def load_checkpoint(self, checkpoint_path) -> Dict[str, Any]:
        if self.load_full_weights and self.zero_stage_3:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.broadcast(checkpoint_path)
            return super().load_checkpoint(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        assert self.lightning_module is not None

        from lightning.pytorch.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        _, client_state = self.deepspeed_engine.load_checkpoint(
            checkpoint_path,
            # Below is the only difference
            load_module_strict=False,
            load_optimizer_states=is_fitting, load_lr_scheduler_states=False
        )
        if client_state is None:
            raise MisconfigurationException(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint "
                "or a single checkpoint file with `Trainer(strategy=DeepSpeedStrategy(load_full_weights=True))`."
            )
        return client_state

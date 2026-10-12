from typing import Literal, Dict, Tuple
import os
import joblib
import fsspec
import torch
import numpy as np
from torch.utils.data import DataLoader
from diffpred.data.mpp_datamodule import MPPDatamodule
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from diffpred.uq.predictive_dist import (
    calibration_plots,
)
from diffpred.trainers import DiffusionTrainer
from diffpred.utils.checkpoint_reload import CheckpointLoader, CheckpointResolver


def _reload_model(trainer, pl_module, criteria: Literal["best", "last"]):
    dm = pl_module.datamodule
    checkpoint_callback = trainer.checkpoint_callback
    reload_path = CheckpointResolver.from_checkpoint_callback(
        checkpoint_callback=checkpoint_callback,
        criteria=criteria,
        prefer_ema=isinstance(pl_module, DiffusionTrainer),
    )
    if not reload_path.is_dir():
        pl_module = CheckpointLoader.load_model(
            model=pl_module,
            checkpoint_path=reload_path,
            datamodule=dm,
            ensure_compatibility=True,
        )
    elif hasattr(pl_module, "set_datamodule"):
        pl_module.set_datamodule(dm)
    return pl_module

class DatamoduleCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # save initial model
        pl_module.set_datamodule(trainer.datamodule)

    def on_validation_start(self, trainer, pl_module):
        # save model before validation
        pl_module.set_datamodule(trainer.datamodule)

    def on_test_start(self, trainer, pl_module):
        # save model after validation
        pl_module.set_datamodule(trainer.datamodule)



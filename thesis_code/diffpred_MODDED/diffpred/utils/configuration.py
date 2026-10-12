from diffpred.callbacks.callbacks import (
    DatamoduleCallback,
)
from diffpred.callbacks.ema import EMA
from diffpred.callbacks.ema_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

class CallbackConfigurer:
    def __init__(self, exp_params, model, datamodule, baseline, trial):
        self.exp_params = exp_params
        self.model = model
        self.datamodule = datamodule
        self.baseline = baseline
        self.trial = trial
        self.all_callbacks = []
        self.checkpoint_callback = None

        self.ema_decay = self.exp_params.get("ema_decay", 0.99)
        self._configure_callbacks()

    def _configure_callbacks(self):
        monitor_metric, self.checkpoint_callback = self.create_checkpoint_callback()
        self._configure_general_callbacks(monitor_metric, self.checkpoint_callback)
        self._configure_diffusion_callbacks()

    # def _configure_general_callbacks(self, monitor_metric, checkpoint_callback):
    #     self.all_callbacks.extend([checkpoint_callback, DatamoduleCallback()])

    #     if self.trial is not None:
    #         self.all_callbacks.append(
    #             PyTorchLightningPruningCallback(self.trial, monitor=monitor_metric)
    #         )
#################################################
    def _configure_general_callbacks(self, monitor_metric, checkpoint_callback):
        self.all_callbacks.extend([checkpoint_callback, DatamoduleCallback()])

        if self.exp_params.get("early_stopping", False):
            self.all_callbacks.append(
                EarlyStopping(
                    monitor=monitor_metric,
                    mode=self.exp_params.monitor_mode,
                    patience=int(self.exp_params.get("early_stopping_patience", 15)),
                    min_delta=float(self.exp_params.get("early_stopping_min_delta", 0.0)),
                )
            )

        if self.trial is not None:
            from optuna_integration import PyTorchLightningPruningCallback

            self.all_callbacks.append(
                PyTorchLightningPruningCallback(self.trial, monitor=monitor_metric)
            )
###################################################

    def _configure_diffusion_callbacks(self):
        if (
            not self.baseline
            and self.model.hparams.get("diffusion_params") is not None
            and self.model.hparams.diffusion_params.diffusion_type == "edm"
        ):
            self.all_callbacks.append(EMA(decay=self.ema_decay))

    def create_checkpoint_callback(self):
        monitor_metric, monitor_mode = (
            self.exp_params.monitor_metric,
            self.exp_params.monitor_mode,
        )
        monitor_metric_name = monitor_metric.replace("val/", "")
        if (
            monitor_metric_name not in (self.model.metric_names)
            and monitor_metric_name != "loss"
        ):
            raise ValueError(
                f"Monitor metric {monitor_metric_name} not in model metric names. Options are {self.model.metric_names} and loss."
            )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.exp_params.output_dir}/checkpoints",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=self.exp_params.save_last_ckpt,
        )

        return monitor_metric, checkpoint_callback

from __future__ import annotations

from dataclasses import dataclass

from hydra_configs.pytorch_lightning.trainer import TrainerConf  # type: ignore


# PyTorch Lightning trainer flags - validation schema & overrides of default values
# See: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
@dataclass
class LightningSettings(TrainerConf):
    # Enable deterministic training
    deterministic: bool = True

    # Number of GPUs to train on
    gpus: int = 1

    # Number of training epochs
    max_epochs: int = 100

    # Progress refresh rate in steps
    progress_bar_refresh_rate: int = 1

    # Validation interval in epochs
    check_val_every_n_epoch: int = 1

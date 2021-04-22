from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from omegaconf.omegaconf import MISSING


# Experiment settings validation schema & default values
@dataclass
class ExperimentSettings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1

    # Path to resume from. Two formats are supported:
    # - local checkpoints: path to checkpoint relative from run (results) directory
    # - wandb artifacts: wandb://ARTIFACT_PATH/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    resume_checkpoint: Optional[str] = None

    # Enable checkpoint saving
    save_checkpoints: bool = True

    # Enable initial validation before training
    validate_before_training: bool = True

    # ----------------------------------------------------------------------------------------------
    # Data loading settings
    # ----------------------------------------------------------------------------------------------
    # Training batch size
    batch_size: int = 32

    # Enable dataset shuffling
    shuffle: bool = True

    # Number of dataloader workers
    num_workers: int = 8

    # ----------------------------------------------------------------------------------------------
    # Dataset specific settings
    # ----------------------------------------------------------------------------------------------
    # PyTorch Lightning datamodule class
    # e.g.: `pl_bolts.datamodules.binary_mnist_datamodule.BinaryMNISTDataModule`
    datamodule: Any = MISSING

    # Shape of input data (channels, height, width)
    input_size: Tuple[int, int, int] = (1, 28, 28)

    # Number of output classes
    n_classes: int = 10

"""
Code template for training neural networks with PyTorch Lightning.

**Features**
- PyTorch Lightning for code organization with datamodules provided by PyTorch Lightning Bolts.
- Experiment configuration handled by Hydra structured configs. This allows for runtime
  config validation and auto-complete support in IDEs.
- Weights & Biases (wandb.ai) logger for metric visualization and checkpoints saving
  as wandb artifacts.
- Console logging and printing with `rich` formatting.
- Typing hints for most of the source code.

**See**
- https://pytorch-lightning.readthedocs.io/en/latest/
- https://lightning-bolts.readthedocs.io/en/latest/
- https://hydra.cc/docs/next/tutorials/intro/
- https://docs.wandb.ai/
- https://github.com/willmcgugan/rich
"""

from __future__ import annotations

import os
from typing import Any, cast

import hydra
import pytorch_lightning as pl
import setproctitle  # type: ignore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from .configs import Config, get_tags, register_configs
from .systems.classifier import ImageClassifier
from .utils.callbacks import CustomCheckpointer, get_resume_checkpoint
from .utils.logging import log
from .utils.rundir import setup_rundir

wandb_logger: WandbLogger


@hydra.main(config_path='configs', config_name='default')
def main(cfg: Config) -> None:
    """
    Main training dispatcher.

    Uses PyTorch Lightning with datamodules provided by PyTorch Lightning Bolts.
    Experiment configuration is handled by Hydra with StructuredConfigs, which allow for config
    validation and provide auto-complete support in IDEs.

    """
    RUN_NAME = os.getenv('RUN_NAME')

    log.info(f'[bold yellow]\\[init] Run name --> {RUN_NAME}')
    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

    run: Run = wandb_logger.experiment  # type: ignore

    # Prepare data using datamodules
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#using-a-datamodule
    datamodule: LightningDataModule = instantiate(
        cfg.experiment.datamodule,
        batch_size=cfg.experiment.batch_size,
        seed=cfg.experiment.seed,
        shuffle=cfg.experiment.shuffle,
        num_workers=cfg.experiment.num_workers
    )

    # Create main system (system = models + training regime)
    system = ImageClassifier(cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    run.tags = tags
    run.notes = str(cfg.notes)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    resume_path = get_resume_checkpoint(cfg, wandb_logger)
    if resume_path is not None:
        log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

    callbacks: list[Any] = []

    checkpointer = CustomCheckpointer(
        period=1,  # checkpointing interval in epochs, but still will save only on validation epoch
        dirpath='checkpoints',
        filename='{epoch}',
    )
    if cfg.experiment.save_checkpoints:
        callbacks.append(checkpointer)

    log.info(f'[bold white]Overriding cfg.pl settings with derived values:')
    log.info(f' >>> resume_from_checkpoint = {resume_path}')
    log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}')
    log.info(f'')

    trainer: pl.Trainer = instantiate(
        cfg.pl,
        logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        resume_from_checkpoint=resume_path,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
    )

    trainer.fit(system, datamodule=datamodule)  # type: ignore
    # Alternative way to call:
    # trainer.fit(system, train_dataloader=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


if __name__ == '__main__':
    setup_rundir()

    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT'),
        entity=os.getenv('WANDB_ENTITY'),
        name=os.getenv('RUN_NAME'),
        save_dir=os.getenv('RUN_DIR'),
    )

    # Init logger from source dir (code base) before switching to run dir (results)
    wandb_logger.experiment  # type: ignore

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()

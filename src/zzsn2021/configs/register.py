from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, cast

import omegaconf
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from omegaconf import SI, DictConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING

# isort: split
from .experiment import ExperimentSettings
from .lightning import LightningSettings
from .optim import OPTIMIZERS, SCHEDULERS, OptimSettings


@dataclass
class Hydra(HydraConf):
    run: RunDir = RunDir("${output_dir}")
    sweep: SweepDir = SweepDir(".", "${output_dir}")


@dataclass
class Config():
    """
    Top-level Hydra config class.
    """
    defaults: List[Any] = field(default_factory=lambda: [
        {'experiment': 'fashion'},
        {'optim': 'adam'},
        {'override hydra/job_logging': 'rich'},
        {'override hydra/hydra_logging': 'rich'},
    ])

    # Path settings
    data_dir: str = SI("${oc.env:DATA_DIR}")
    output_dir: str = SI("${oc.env:RUN_DIR}")

    # Runtime configuration
    hydra: Hydra = Hydra()
    pl: LightningSettings = LightningSettings()

    # Experiment settings --> experiment/*.yaml
    experiment: ExperimentSettings = MISSING

    # Optimizer & scheduler settings --> optim/*.yaml
    optim: OptimSettings = MISSING

    # wandb metadata
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


def register_configs():
    """
    Register configuration options in the main ConfigStore.instance().

    The term `config` is used for a StructuredConfig at the root level (normally switchable with `-cn`
    flag in Hydra, here we use only one default config). Fields of the main config use StructuredConfigs
    with class names ending in `Settings`. `Conf` suffix is used for external schemas provided by
    the `hydra-torch` package for PyTorch/PyTorch Lightning integration, e.g. `AdamConf`.
    """
    cs = ConfigStore.instance()

    # Main config
    cs.store(name='default', node=DictConfig(Config()))

    # Config groups with defaults, YAML files validated by Python structured configs
    # e.g.: `python -m zzsn2021.main experiment=fashion`
    cs.store(group='experiment', name='schema_experiment', node=ExperimentSettings)
    cs.store(group='optim', name='schema_optim', node=OptimSettings)

    # Specific schemas, YAML files should inherit them as a default, e.g:
    # defaults:
    #   - schema_optim
    #   - schema_optim_adam
    for key, node in OPTIMIZERS.items():
        name = f'schema_optim_{key}'
        cs.store(group='optim', name=name, node=node, package='optim.optimizer')
    for key, node in SCHEDULERS.items():
        name = f'schema_optim_lrscheduler_{key}'
        cs.store(group='optim', name=name, node=node, package='optim.scheduler')


def _get_tags(cfg: dict[str, Any]) -> Iterator[str]:
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from _get_tags(cast(Dict[str, Any], value))
        if key == '_tags_':
            if isinstance(value, list):
                for v in cast(List[str], value):
                    yield v
            else:
                if value is not None:
                    value = cast(str, value)
                    yield value


def get_tags(cfg: DictConfig):
    """
    Extract all tags from a nested DictConfig object.
    """
    cfg_dict = cast(Dict[str, Any], omegaconf.OmegaConf.to_container(cfg, resolve=True))
    if 'tags' in cfg_dict:
        cfg_dict['_tags_'] = cfg_dict['tags']
    return list(_get_tags(cfg_dict))

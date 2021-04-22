from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# isort: off
from hydra_configs.torch.optim.adam import AdamConf  # type: ignore
from hydra_configs.torch.optim.sgd import SGDConf  # type: ignore
from hydra_configs.torch.optim.lr_scheduler import ExponentialLRConf  # type: ignore
from hydra_configs.torch.optim.lr_scheduler import MultiStepLRConf  # type: ignore
from hydra_configs.torch.optim.lr_scheduler import StepLRConf  # type: ignore
# isort: on
from omegaconf import MISSING

# Schema for config validation
OPTIMIZERS = {
    'adam': AdamConf,
    'sgd': SGDConf,
    # ...
}

SCHEDULERS = {
    'exponentiallr': ExponentialLRConf,
    'steplr': StepLRConf,
    'multisteplr': MultiStepLRConf,
    # ...
}


@dataclass
class OptimSettings:
    optimizer: Any = MISSING
    scheduler: Optional[Any] = MISSING

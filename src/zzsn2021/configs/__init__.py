# Using redundant module aliases for public export
# https://github.com/microsoft/pyright/blob/master/docs/typed-libraries.md#library-interface
from __future__ import annotations

from .register import Config as Config
from .register import get_tags as get_tags
from .register import register_configs as register_configs

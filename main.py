"""
Main file. Entrypoint for all experiments and the one that parses
the `hydra` configuration files.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from helpers.derive_config import derive_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


# Add custom handy resolvers
OmegaConf.register_new_resolver(
    "env",
    lambda key: {"prefix": Path(sys.executable).parent, "cwd": os.getcwd()}.get(key),
)

# Support numeric interpolation used in config files, e.g. `${sum:${dataset.theta_dim},4}`.
if not OmegaConf.has_resolver("sum"):
    OmegaConf.register_new_resolver("sum", lambda *values: sum(values))

if not OmegaConf.has_resolver("prod"):
    OmegaConf.register_new_resolver(
        "prod", lambda *values: math.prod(int(v) for v in values)
    )


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig) -> None:
    """Parses configuration object, derives fields and passes the final
    config to the selected launcher

    :param cfg: CL-specified configuration object
    :type cfg: DictConfig
    """
    cfg = derive_config(cfg)
    instantiate(cfg.launcher)(cfg=cfg)


if __name__ == "__main__":
    main()

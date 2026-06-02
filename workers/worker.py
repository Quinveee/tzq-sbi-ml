"""Contains the main worker function to run all experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

# The worker spawns a new python process, so any needed resolvers from now on
# must be registerd here (doing that in main.py won't work)
if not OmegaConf.has_resolver("sum"):
    OmegaConf.register_new_resolver("sum", lambda *values: sum(values))

if not OmegaConf.has_resolver("prod"):
    import math as _math

    OmegaConf.register_new_resolver(
        "prod", lambda *values: _math.prod(int(v) for v in values)
    )

# `${env:cwd}` is used in the dataset configs to point at the repo-local data
# dir. run_snellius.sh cd's into the project root before launching the worker,
# so os.getcwd() resolves to the same path the launcher used at submit time.
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env",
        lambda key: {"prefix": Path(sys.executable).parent, "cwd": os.getcwd()}.get(
            key
        ),
    )


def run(cfg: DictConfig) -> None:
    """
    Worker function.

    :param cfg: Final configuration object for the experiment
    :type cfg: DictConfig
    """
    print(
        f"exp={cfg.exp.key} model={cfg.model.key} dataset={cfg.dataset.key} run={cfg.data.run}"
    )
    instantiate(cfg.exp)(cfg=cfg)()


if __name__ == "__main__":
    run(OmegaConf.load(sys.argv[1]))

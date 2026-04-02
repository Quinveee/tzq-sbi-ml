"""Contains the main worker function to run all experiments."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

# The worker spawns a new python process, so any needed resolvers from now on
# must be registerd here (doing that in main.py won't work)
if not OmegaConf.has_resolver("sum"):
    OmegaConf.register_new_resolver("sum", lambda *values: sum(values))


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

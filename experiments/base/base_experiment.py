"""
Base experiment class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from ..logger import LOGGER as _LOGGER
from ..plotting import plot_llr
from .schemas import Chekcpoints

if TYPE_CHECKING:
    from omegaconf import DictConfig


LOGGER = _LOGGER.getChild(__name__)


class BaseExperiment(ABC):
    """
    Base experiment class defining abstract methods
    """

    def __init__(self, cfg: DictConfig, key: Optional[str] = None):
        self._key = key
        self.cfg = cfg
        self.checkpoints: Optional[Chekcpoints] = None

    def __call__(self):
        return self.run()

    def __str__(self):
        return self._key if self._key else ""

    @abstractmethod
    def _init(self):
        pass

    @abstractmethod
    def _run(self):
        pass

    def init(self) -> None:
        """
        Just creating run dir if needed and getting
        checkpoints file if present
        """
        self.cfg.data.run_dir = self.init_run_dir()
        self.checkpoints = self.init_checkpoints()

    def init_run_dir(self) -> Path:
        """
        Create run directory if not existent yet

        :return: Run dir path
        :rtype: Path
        """
        run_dir = Path(self.cfg.data.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def init_checkpoints(self) -> Chekcpoints:
        """
        Load checkpoints file if present

        :return: Checkpoints object (may be empty)
        :rtype: Chekcpoints
        """
        ckpts_path = self.cfg.data.run_dir / self.cfg.data.ckpts
        if ckpts_path.exists():
            LOGGER.info(f"Found existing checkpoints at {ckpts_path}, loading ...")
            return Chekcpoints(
                **torch.load(ckpts_path, map_location="cpu", weights_only=False)
            )
        LOGGER.info(f"Checkpoints file not found.")
        return Chekcpoints()

    def save_checkpoints(self) -> None:
        """
        Save current state to checkpoints file

        """
        ckpts_path = self.cfg.data.run_dir / self.cfg.data.ckpts
        torch.save(asdict(self.checkpoints), ckpts_path)

    def plot(self, label: str = ""):
        plot_llr(
            llr_list=[self.checkpoints.limits.llr],
            grid=self.checkpoints.limits.grid,
            param_names=self.checkpoints.limits.param_names,
            ranges=self.checkpoints.limits.ranges,
            resolutions=self.checkpoints.limits.resolutions,
            labels=[label],
            to=Path(self.cfg.data.run_dir) / "llr.png",
        )

    def run(self) -> None:
        """
        Main run method called to run experiments
        Specific functionality is implemented by defining the
        _init() and _run() methods in subclasses

        """
        # Init base
        self.init()

        # Init subclass
        self._init()

        try:
            # Run subclass
            self._run()
        finally:
            # Save results
            self.save_checkpoints()

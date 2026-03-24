"""
Preprocessing experiment class for particles
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base.preprocessing import BasePreprocessing

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PreprocessingParticles(BasePreprocessing):
    """
    Preprocessing experiment for particle data.
    Handles data transformation and feature extraction.
    """

    def __init__(self, cfg: DictConfig, key: str = "preprocessing") -> None:
        super().__init__(cfg=cfg, key=key)

    def __call__(self) -> None:
        """Execute the preprocessing pipeline"""
        self.preprocess()

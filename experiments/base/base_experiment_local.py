"""
Base experiment class for score regression (local) experiments
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..limits import AsymptoticLimitsHistos
from ..logger import LOGGER as _LOGGER
from .base_experiment_ml import BaseExperimentML
from .schemas import ModelOutput, PredictionOutput, RawData, TargetOutput

if TYPE_CHECKING:
    import torch

LOGGER = _LOGGER.getChild(__name__)


class BaseExperimentLocal(BaseExperimentML):
    """
    Base experiment class for score regression experiments
    """

    asymptotics_cls = AsymptoticLimitsHistos

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "local"
        super().__init__(*args, **kwds)

    def _preds(self, *args, **kwds):
        pass

    def _load_raw_data(self, source: str) -> RawData:
        """
        Load raw data from root path `source` assuming Madminer
        naming conventions

        :param source: Root path for the numpy files
        :type source: str
        :return: Container with the loaded numpy arrays
        :rtype: RawData
        """
        # Get source
        source = Path(source)

        # Get max samples
        max_samples = self.cfg.train.get("clamp_samples", None)

        # Load raw data
        x_train = np.load(source / "x_train_score.npy")[:max_samples]
        x_test = np.load(source / "x_test.npy")

        # Load train/test labels
        score_train = np.load(source / "t_xz_train_score.npy")[:max_samples]
        score_test = np.load(source / "t_xz_test_score.npy")

        return RawData(
            x_train=x_train,
            score_train=score_train,
            x_test=x_test,
            score_test=score_test,
        )

    def _load_dataset(
        self, raw: RawData, mode: Literal["train", "test"] = "train"
    ) -> torch.utils.data.Dataset:
        """
        Load train or test datasets

        :param raw: Container with the loaded numpy arrays
        :type raw: ParametrizedRawData
        :param mode: What partition to use
        :type mode: Literal["train", "test"]
        :return: Torch dataset
        :rtype: Dataset
        """
        if mode == "train":
            return self.dataset_cls(x=raw.x_train, score=raw.score_train)
        elif mode == "test":
            return self.dataset_cls(x=raw.x_test, score=raw.score_test)
        raise ValueError(f"Invalid mode {mode}")

    def _eval(self, output: ModelOutput) -> torch.Tensor:
        """
        Select the predicted score from output container

        :param output: Packed model output and targets
        :type output: ModelOutput
        :return: The predicted score
        :rtype: Tensor
        """

        return output.pred.score

    def pack_output(self, score_pred: torch.Tensor, score: torch.Tensor) -> ModelOutput:
        """
        Pack model output and target in one container

        :param score_pred: Predicted score
        :type score_pred: torch.Tensor
        :param score: Target (true) score
        :type score: torch.Tensor
        :return: Container with model output and target
        :rtype: ModelOutput
        """
        return ModelOutput(PredictionOutput(score=score_pred), TargetOutput(score))

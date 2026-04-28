"""
Base class for all likelihood-ratio regression experiments
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.autograd import grad

from ..limits import AsymptoticLimitsRatios
from .base_experiment_ml import BaseExperimentML
from .schemas import (
    ModelOutput,
    ParametrizedPredictionOutput,
    ParametrizedRawData,
    ParametrizedTargetOutput,
)


class BaseExperimentRatios(BaseExperimentML):
    """
    Base class for ML likelihood ratio regression
    """

    asymptotics_cls = AsymptoticLimitsRatios

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "ratios"
        super().__init__(*args, **kwds)
        self._val_log_r: list[torch.Tensor] = []
        self._val_y: list[torch.Tensor] = []

    def _preds(self, *args, **kwargs) -> ...: ...

    def _val_extra_init(self) -> None:
        self._val_log_r = []
        self._val_y = []

    def _val_extra_accumulate(self, output: ModelOutput) -> None:
        log_r = getattr(output.pred, "log_ratio", None)
        label = getattr(output.target, "label", None)
        if log_r is None or label is None:
            return
        self._val_log_r.append(log_r.detach().reshape(-1).cpu())
        self._val_y.append(label.detach().reshape(-1).cpu())

    def _val_extra_finalize(self) -> dict:
        """Expected Calibration Error of p(y=1|x)=sigmoid(log r̂)."""
        if not self._val_log_r:
            return {}
        log_r = torch.cat(self._val_log_r).float()
        y = torch.cat(self._val_y).float()
        if y.numel() == 0:
            return {}
        # Skip when labels carry no signal (e.g. an all-zero placeholder).
        if (y.unique().numel() < 2):
            return {}

        p = torch.sigmoid(log_r).clamp(0.0, 1.0)
        n_bins = 15
        edges = torch.linspace(0.0, 1.0, n_bins + 1)
        n_total = p.numel()
        ece = 0.0
        for i in range(n_bins):
            lo = edges[i].item()
            hi = edges[i + 1].item()
            mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
            n_b = int(mask.sum().item())
            if n_b == 0:
                continue
            ece += (n_b / n_total) * abs(p[mask].mean().item() - y[mask].mean().item())
        return {"calibration_error": float(ece)}

    def _load_raw_data(self, source: str) -> ParametrizedRawData:
        """
        Load raw data from root path `source` assuming Madminer
        naming conventions

        :param source: Root path for the numpy files
        :type source: str
        :return: Container with the loaded numpy arrays
        :rtype: ParametrizedRawData
        """
        # Get source
        source = Path(source)

        # Get max samples
        max_samples = self.cfg.train.get("clamp_samples", None)

        # Load train data
        x_train = np.load(source / "x_train_ratio.npy")[:max_samples]
        theta_train = np.load(source / "theta0_train_ratio.npy")[:max_samples]

        # Load train labels
        ratio_train = np.load(source / "r_xz_train_ratio.npy")[:max_samples]
        score_train = np.load(source / "t_xz_train_ratio.npy")[:max_samples]
        labels_train = np.load(source / "y_train_ratio.npy")[:max_samples]

        # Load test data
        # NOTE: ratio targets (r_xz, t_xz, y) were generated for events in
        # x_test_ratio.npy at the per-event sampled θ₀ in theta0_test_ratio.npy.
        # theta_test.npy is a separate (all-zero) SM sample used for Asimov, and
        # is NOT aligned with the ratio labels.
        x_test = np.load(source / "x_test_ratio.npy")
        theta_test = np.load(source / "theta0_test_ratio.npy")

        # Load test labels 
        ratio_test = np.load(source / "r_xz_test_ratio.npy")
        score_test = np.load(source / "t_xz_test_ratio.npy")
        labels_test = np.load(source / "y_test_ratio.npy")

        return ParametrizedRawData(
            x_train=x_train,
            theta_train=theta_train,
            ratio_train=ratio_train,
            score_train=score_train,
            labels_train=labels_train,
            x_test=x_test,
            theta_test=theta_test,
            ratio_test=ratio_test,
            score_test=score_test,
            labels_test=labels_test,
        )

    def _load_dataset(
        self, raw: ParametrizedRawData, mode: Literal["train", "test"] = "train"
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
            return self.dataset_cls(
                x=raw.x_train,
                theta=raw.theta_train,
                score=raw.score_train,
                ratio=raw.ratio_train,
                labels=raw.labels_train,
            )
        if mode == "test":
            return self.dataset_cls(
                x=raw.x_test,
                theta=raw.theta_test,
                score=raw.score_test,
                ratio=raw.ratio_test,
                labels=raw.labels_test,
            )
        raise ValueError(f"Invalid mode {mode}")

    # NOTE; In possible future in which I calculate loss on test data,
    # i might need access to the score here depending on the loss function used
    def _eval(self, output: ModelOutput) -> torch.Tensor:
        """
        Return the predicted log lkelihood ratio

        :param output: Container with model output tensors and targets
        :type output: ModelOutput
        :return: Predicted log likelihood ratio
        :rtype: torch.Tensor
        """
        return output.pred.log_ratio

    def pack_output(
        self,
        theta: torch.Tensor,
        log_ratio_pred: torch.Tensor,
        score: torch.Tensor,
        ratio: torch.Tensor,
        label: torch.Tensor,
    ) -> ModelOutput:
        """
        Pack model output (both predictions and targets)
        in a container and optionally calculate the
        gradient of the output with respect to model parameters,
        e.g., the score

        :param theta: Theory parameters tensor
        :type theta: torch.Tensor
        :param log_ratio_pred: Log likelihood ratio prediction
        :type log_ratio_pred: torch.Tensor
        :param score: True score
        :type score: torch.Tensor
        :param ratio: True likelihood ratio
        :type ratio: torch.Tensor
        :param label: True label
        :type label: torch.Tensor
        :return: Model output and targets packed in one container
        :rtype: ModelOutput
        """
        score_pred = None
        if self.loss_fn.REQUIRES_SCORE:
            (score_pred,) = grad(
                log_ratio_pred,
                theta,
                grad_outputs=torch.ones_like(log_ratio_pred),
                only_inputs=True,
                create_graph=True,
            )
        return ModelOutput(
            pred=ParametrizedPredictionOutput(
                score=score_pred, log_ratio=log_ratio_pred
            ),
            target=ParametrizedTargetOutput(score=score, ratio=ratio, label=label),
        )

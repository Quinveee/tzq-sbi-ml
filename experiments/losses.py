"""
Loss functions implemented in Madminer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .base.schemas import ModelOutput

# NOTE: The `REQUIRES_SCORE` flag is `True` only when the loss function
# depends on the score and it is to be used with a log-likelihood ratio
# regressor (e.g. SALLY uses the score but is to be used with score regressors)


class Loss(ABC):
    """
    Base Loss class. The `REQUIRES_SCORE` flag is `True` only when the loss
    function depends on the score and it is to be used with a log-likelihood ratio
    regressor
    """

    REQUIRES_SCORE: bool

    def __init__(self, **kwds):
        self._params = kwds

    @classmethod
    @abstractmethod
    def forward(cls, output: ModelOutput, **kwds):
        pass

    def __call__(self, output):
        return self.forward(output=output, **self._params)


class ALICES(Loss):
    REQUIRES_SCORE = True

    @classmethod
    def forward(cls, output: ModelOutput, **kwds):
        alpha = kwds.get("lambda_score", 1.0)
        s = torch.sigmoid(-torch.log(output.target.ratio))
        bce = F.binary_cross_entropy_with_logits(-output.pred.log_ratio, s)
        if alpha > 0:
            mse = F.mse_loss(
                (1.0 - output.target.label) * output.pred.score,
                (1.0 - output.target.label) * output.target.score,
            )
            return bce + alpha * mse
        return bce


class ALICE(Loss):
    REQUIRES_SCORE = False

    @classmethod
    def forward(cls, output, **kwds):
        return ALICES.forward(output=output, alpha=0.0)


class SALLY(Loss):
    REQUIRES_SCORE = False

    @classmethod
    def forward(cls, output, **kwds):
        return F.mse_loss(output.pred.score, output.target.score)


class CARL(Loss):
    REQUIRES_SCORE = False

    @classmethod
    def forward(cls, output, **kwds):
        return F.binary_cross_entropy_with_logits(
            -output.pred.log_ratio, output.target.label
        )


class ROLR(Loss):
    REQUIRES_SCORE = False

    @classmethod
    def forward(cls, output, **kwds):
        log_r_clip = kwds.get("log_r_clip", 10.0)
        ratio = torch.clamp(
            output.target.ratio, np.exp(-log_r_clip), np.exp(log_r_clip)
        )
        log_ratio_pred = torch.clamp(output.pred.log_ratio, -log_r_clip, log_r_clip)
        loss_inv = F.mse_loss(
            (1.0 - output.target.label) * torch.exp(-log_ratio_pred),
            (1.0 - output.target.label) * (1.0 / ratio),
        )
        loss = F.mse_loss(
            output.target.label * torch.exp(log_ratio_pred), output.target.label * ratio
        )
        return loss + loss_inv


class NLL(Loss):
    """Negative log-likelihood loss for conditional density estimation.

    Expects ``output.pred.log_ratio`` to carry the per-event log-probability
    ``log p(x | theta)`` produced by a conditional flow. Trained on numerator
    samples only (label == 0) by default, since those are the events drawn
    from p(x | theta).
    """

    REQUIRES_SCORE = False

    @classmethod
    def forward(cls, output, **kwds):
        numerator_only = kwds.get("numerator_only", True)
        log_p = output.pred.log_ratio.reshape(-1)
        if numerator_only and output.target.label is not None:
            mask = (output.target.label.reshape(-1) < 0.5).to(log_p.dtype)
            denom = mask.sum().clamp_min(1.0)
            return -(mask * log_p).sum() / denom
        return -log_p.mean()


# Alices but with Rolr-style score regularization, to be used with log-likelihood ratio regressors
class JointLoss(Loss):
    REQUIRES_SCORE = True

    @classmethod
    def forward(cls, output, **kwds):
        lambda_score = kwds.get("lambda_score", 1.0)
        log_r_clip = kwds.get("log_r_clip", 10.0)

        l_ratio = ROLR.forward(output, log_r_clip=log_r_clip)

        # TODO: maybe add score normalization to stabalize the score further
        l_score = F.mse_loss(
            (1.0 - output.target.label) * output.pred.score,
            (1.0 - output.target.label) * output.target.score,
        )

        return l_ratio + lambda_score * l_score
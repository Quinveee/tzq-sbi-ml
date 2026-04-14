"""LorentzNet architecture wrappers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from .base_wrapper import BaseWrapper
from models.lorentznet import normsq4, psi


def _particle_mass_scalar(particles: torch.Tensor) -> torch.Tensor:
    """Per-particle Lorentz-invariant scalar feature (psi of Minkowski norm)."""
    x = particles[:, -4:] if particles.shape[-1] > 4 else particles
    return psi(normsq4(x)).unsqueeze(-1)


def _broadcast_event_features(
    features: Optional[torch.Tensor], ptr: torch.Tensor
) -> Optional[torch.Tensor]:
    """Broadcast event-level features (B, D) to particle-level (N, D)."""
    if features is None:
        return None
    if features.ndim == 1:
        features = features.unsqueeze(-1)
    if features.shape[-1] == 0:
        return None
    ptr_long = ptr.to(dtype=torch.long)
    return features.repeat_interleave(ptr_long[1:] - ptr_long[:-1], dim=0)


class BaseLorentzNetWrapper(BaseWrapper, ABC):
    """Base LorentzNet wrapper"""

    def __init__(self, *args, **kwds):
        kwds["key"] = "LorentzNet"
        super().__init__(*args, **kwds)

    @abstractmethod
    def embed(self, particles: torch.Tensor, **kwds) -> torch.Tensor:
        pass

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Forward wrapper for LorentzNet.

        :param particles: Flattened particle four-momenta (possibly with leading
            conditioning channels which are ignored for the coordinate input).
        :type particles: torch.Tensor
        :param ptr: Event pointer
        :type ptr: torch.Tensor
        :param force_math: Unused; kept for signature parity with other wrappers.
        :param embedding_kwargs: Extra kwargs forwarded to the `embed` method.
        """
        embedding_kwargs = dict(embedding_kwargs or {})
        embedding_kwargs.pop("ptr", None)
        x = particles[:, -4:] if particles.shape[-1] > 4 else particles
        scalars = self.embed(particles, ptr=ptr, **embedding_kwargs)
        return self.net(scalars=scalars, x=x, ptr=ptr)


class LocalLorentzNetWrapper(BaseLorentzNetWrapper):
    def embed(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        preprocessed: Optional[torch.Tensor] = None,
        met: Optional[torch.Tensor] = None,
        **kwds,
    ) -> torch.Tensor:
        """Per-particle invariant-mass scalar optionally augmented with
        event-level preprocessed features and/or MET broadcast per particle.
        """
        scalars = [_particle_mass_scalar(particles)]
        pp = _broadcast_event_features(preprocessed, ptr)
        if pp is not None:
            scalars.append(pp)
        m = _broadcast_event_features(met, ptr)
        if m is not None:
            scalars.append(m)
        return torch.cat(scalars, dim=-1)


class ParametrizedLorentzNetWrapper(BaseLorentzNetWrapper):
    def embed(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        ptr: torch.Tensor,
        preprocessed: Optional[torch.Tensor] = None,
        met: Optional[torch.Tensor] = None,
        **kwds,
    ) -> torch.Tensor:
        """Concatenate theta (broadcast per particle) with the mass scalar and
        optional preprocessed / MET event-level features broadcast per particle.
        """
        ptr_long = ptr.to(dtype=torch.long)
        scalars = [
            _particle_mass_scalar(particles),
            theta.repeat_interleave(ptr_long[1:] - ptr_long[:-1], dim=0),
        ]
        pp = _broadcast_event_features(preprocessed, ptr)
        if pp is not None:
            scalars.append(pp)
        m = _broadcast_event_features(met, ptr)
        if m is not None:
            scalars.append(m)
        return torch.cat(scalars, dim=-1)

"""LorentzNet architecture wrappers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from .base_wrapper import BaseWrapper
from .utils import derive_valid_mask
from models.lorentznet import normsq4, psi


def _particle_mass_scalar(particles: torch.Tensor) -> torch.Tensor:
    """Per-particle Lorentz-invariant scalar feature (psi of Minkowski norm).
    Works on padded (B, L, D) and flat (N, D) shapes; only the last-4 channels
    are read as the four-momentum."""
    x = particles[..., -4:] if particles.shape[-1] > 4 else particles
    return psi(normsq4(x)).unsqueeze(-1)


def _broadcast_event_features(
    features: Optional[torch.Tensor], L_max: int
) -> Optional[torch.Tensor]:
    """Broadcast event-level (B, F) features to per-token (B, L_max, F)."""
    if features is None:
        return None
    if features.ndim == 1:
        features = features.unsqueeze(-1)
    if features.shape[-1] == 0:
        return None
    if features.ndim == 3:
        return features
    return features.unsqueeze(1).expand(-1, L_max, -1)


class BaseLorentzNetWrapper(BaseWrapper, ABC):
    """Base LorentzNet wrapper — operates on padded (B, L_max, ...) particles."""

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
        valid_mask: Optional[torch.Tensor] = None,
        force_math: bool = False,
        embedding_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward over padded particles ``(B, L_max, 4)`` with a
        ``(B, L_max)`` validity mask. ``ptr`` is accepted for compatibility
        with call sites that still hand it in and is used only to derive
        ``valid_mask`` if it wasn't passed explicitly."""
        embedding_kwargs = dict(embedding_kwargs or {})
        embedding_kwargs.pop("ptr", None)
        embedding_kwargs.pop("valid_mask", None)

        if particles.ndim != 3:
            raise ValueError(
                "LorentzNet wrapper expects particles of shape (B, L_max, 4); "
                f"got {tuple(particles.shape)}."
            )
        L_max = particles.shape[1]
        if valid_mask is None:
            valid_mask = derive_valid_mask(ptr, L_max).to(particles.device)
        else:
            valid_mask = valid_mask.to(torch.bool)

        x = particles[..., -4:] if particles.shape[-1] > 4 else particles
        scalars = self.embed(particles, valid_mask=valid_mask, **embedding_kwargs)
        return self.net(scalars=scalars, x=x, valid_mask=valid_mask)


class LocalLorentzNetWrapper(BaseLorentzNetWrapper):
    def embed(
        self,
        particles: torch.Tensor,
        valid_mask: torch.Tensor,
        preprocessed: Optional[torch.Tensor] = None,
        met: Optional[torch.Tensor] = None,
        **kwds,
    ) -> torch.Tensor:
        L_max = particles.shape[1]
        scalars = [_particle_mass_scalar(particles)]
        pp = _broadcast_event_features(preprocessed, L_max)
        if pp is not None:
            scalars.append(pp)
        m = _broadcast_event_features(met, L_max)
        if m is not None:
            scalars.append(m)
        return torch.cat(scalars, dim=-1)


class ParametrizedLorentzNetWrapper(BaseLorentzNetWrapper):
    def embed(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        valid_mask: torch.Tensor,
        preprocessed: Optional[torch.Tensor] = None,
        met: Optional[torch.Tensor] = None,
        **kwds,
    ) -> torch.Tensor:
        L_max = particles.shape[1]
        scalars = [
            _particle_mass_scalar(particles),
            _broadcast_event_features(theta, L_max),
        ]
        pp = _broadcast_event_features(preprocessed, L_max)
        if pp is not None:
            scalars.append(pp)
        m = _broadcast_event_features(met, L_max)
        if m is not None:
            scalars.append(m)
        return torch.cat(scalars, dim=-1)

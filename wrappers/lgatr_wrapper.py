"""
LGATr wrappers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Literal, Optional

import torch
from hydra.utils import instantiate
from lgatr import extract_scalar
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .decorators import filter_empty_tensor_warning
from .embed import to_multivector
from .utils import derive_valid_mask, get_backends

if TYPE_CHECKING:
    import torch.nn as nn
    from omegaconf import DictConfig
    from torch import Tensor


class BaseLGATrWrapper(BaseWrapper, ABC):
    """Base LGATr wrapper"""

    def __init__(self, *args, mode: Literal["tokens", "channels"] = "channels", **kwds) -> None:
        kwds["key"] = "LGATr"
        super().__init__(*args, **kwds)
        self.mode = mode
        self.net = self.init_net(self.net)

    @filter_empty_tensor_warning
    def init_net(self, net: DictConfig) -> nn.Module:
        """
        Inits the LGATr module according to its configuration object

        :param net: Config description of the LGATr module
        :type net: DictConfig
        :return: Torch LGATr module
        :rtype: Module
        """
        return instantiate(net)

    @abstractmethod
    def output(self, multivectors, valid_mask, scalars=None):
        pass

    @abstractmethod
    def embed_mv(self, *args, **kwargs):
        pass

    @staticmethod
    def _broadcast_event_scalars(
        scalars: Optional[Tensor], L_max: int
    ) -> Optional[Tensor]:
        """Broadcast event-level (B, F) scalars to per-token (B, L_max, F)."""
        if scalars is None:
            return None
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(-1)
        if scalars.ndim == 3:
            return scalars
        if scalars.ndim != 2:
            raise ValueError(
                f"Expected scalars (B, F), (B, L, F), or 1D; got {scalars.shape}"
            )
        if scalars.shape[-1] == 0:
            return None
        return scalars.unsqueeze(1).expand(-1, L_max, -1)

    def forward(
        self,
        particles: Tensor,
        ptr: Tensor,
        valid_mask: Optional[Tensor] = None,
        scalars: Optional[Tensor] = None,
        met: Optional[Tensor] = None,
        force_math: bool = False,
        embedding_kwargs: Optional[Dict] = None,
    ) -> Tensor:
        """
        Forward for any LGATr wrapper. Operates on padded inputs:

            particles:  (B, L_max, 4)
            valid_mask: (B, L_max) — True for real tokens, False for padding.

        LGATr accepts ``multivectors`` of shape ``(..., items, c, 16)``, so we
        pass ``(B, L_max, c, 16)`` directly and use a ``(B, 1, 1, L_max)``
        key-only attention mask to suppress padding contributions.
        """
        backends = get_backends(force_math)

        if embedding_kwargs is None:
            embedding_kwargs = {}
        else:
            embedding_kwargs = dict(embedding_kwargs)

        if particles.ndim != 3:
            raise ValueError(
                "LGATr wrapper expects particles of shape (B, L_max, 4); "
                f"got {tuple(particles.shape)}."
            )
        B, L_max = particles.shape[0], particles.shape[1]
        if valid_mask is None:
            valid_mask = derive_valid_mask(ptr, L_max).to(particles.device)
        else:
            valid_mask = valid_mask.to(torch.bool)

        mode = embedding_kwargs.get("mode", self.mode)
        if mode != "channels":
            raise ValueError(
                "Padded LGATr wrapper supports mode='channels' only; the "
                "tokens-mode theta layout has not been ported to padded data."
            )
        theta = embedding_kwargs.get("theta", None)

        # Scalar channels carry, in order: θ | preprocessed | MET — all
        # broadcast across the per-token axis.
        scalar_parts = []
        ts = self._broadcast_event_scalars(theta, L_max)
        if ts is not None:
            scalar_parts.append(ts)
        ps = self._broadcast_event_scalars(scalars, L_max)
        if ps is not None:
            scalar_parts.append(ps)
        ms = self._broadcast_event_scalars(met, L_max)
        if ms is not None:
            scalar_parts.append(ms)
        scalars_in = torch.cat(scalar_parts, dim=-1) if scalar_parts else None

        # Embed four-momenta into multivectors (B, L_max, c, 16).
        mv = self.embed_mv(particles, **embedding_kwargs)

        # Key-only padding mask: True means "attend"; (B, 1, 1, L) broadcasts
        # across LGATr's internal heads and queries.
        attn_mask = valid_mask[:, None, None, :]

        with sdpa_kernel(backends):
            out_mv, out_s = self.net(
                multivectors=mv, scalars=scalars_in, attn_mask=attn_mask
            )

        return self.output(multivectors=out_mv, scalars=out_s, valid_mask=valid_mask)


class LocalLGATrWrapper(BaseLGATrWrapper):

    def embed_mv(self, particles: Tensor, theta_dim: int, **kwargs) -> Tensor:
        """
        For the local experiment, just repeat the embedded multivector
        `theta_dim` times along the multivector channel dimension

        :param particles: Particles fourmomenta
        :type particles: Tensor
        :param theta_dim: Theory parameter dimension
        :type theta_dim: int
        :return: Multivectors
        :rtype: Tesor

        """
        # to_multivector handles padded (B, L_max, 4) -> (B, L_max, 1, 16).
        return to_multivector(particles).repeat(1, 1, theta_dim, 1)

    def output(
        self, multivectors: Tensor, valid_mask: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """Masked-mean over real tokens, then extract the scalar component.

        ``multivectors`` has shape ``(B, L_max, C, 16)``; output is
        ``(B, C)`` — the mean is restricted to valid tokens via valid_mask.
        """
        weight = valid_mask.to(multivectors.dtype)[..., None, None]  # (B, L, 1, 1)
        counts = weight.sum(dim=1).clamp(min=1)  # (B, 1, 1)
        pooled = (multivectors * weight).sum(dim=1) / counts  # (B, C, 16)
        return extract_scalar(pooled)[..., 0]  # (B, C)


class ParametrizedLGATrWrapper(BaseLGATrWrapper):
    def embed_mv(
        self,
        particles: Tensor,
        theta: Tensor,
        mode: Literal["tokens", "channels"] = "channels",
        **kwargs,
    ) -> Tensor:
        """
        Embed particle 4-momenta into multivectors. θ is **not** placed in the
        multivector pathway — it is Lorentz-invariant and routed through the
        scalar channels instead (see ``BaseLGATrWrapper.forward``). Tokens-mode
        theta is not supported on the padded path.
        """
        if mode != "channels":
            raise ValueError(
                "Padded ParametrizedLGATrWrapper supports mode='channels' only."
            )
        return to_multivector(particles)  # (B, L_max, 1, 16)

    def output(
        self, multivectors: Tensor, valid_mask: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """Masked-mean over real tokens; return the scalar component of the
        first output multivector channel as the regressed log-likelihood ratio."""
        weight = valid_mask.to(multivectors.dtype)[..., None, None]
        counts = weight.sum(dim=1).clamp(min=1)
        pooled = (multivectors * weight).sum(dim=1) / counts  # (B, C, 16)
        return extract_scalar(pooled)[..., :1, 0]  # (B, 1)

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
from .utils import att_mask, get_backends, ptr2index

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
    def output(self, multivectors, index, scalars=None):
        pass

    @abstractmethod
    def embed_mv(self, *args, **kwargs):
        pass

    @staticmethod
    def _expand_event_scalars(
        scalars: Optional[Tensor],
        ptr: Tensor,
        *,
        mode: Literal["tokens", "channels"],
        theta_dim: int,
    ) -> Optional[Tensor]:
        """
        Expand event-level scalar features from shape (batch, s_dim) to token-level
        shape (1, n_tokens, s_dim), matching LGATr's flattened token representation.
        """
        if scalars is None:
            return None

        if scalars.ndim == 3:
            return scalars

        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(-1)

        if scalars.ndim != 2:
            raise ValueError(
                f"Expected scalars with shape (batch, s_dim) or (1, n_tokens, s_dim), got {scalars.shape}"
            )

        ptr = ptr.long()
        repeats = ptr[1:] - ptr[:-1]
        if mode == "tokens":
            repeats = repeats + theta_dim

        expanded = scalars.repeat_interleave(repeats, dim=0)
        return expanded.unsqueeze(0)

    @staticmethod
    def _expand_theta_scalars(
        theta: Optional[Tensor],
        ptr: Tensor,
        *,
        mode: Literal["tokens", "channels"],
    ) -> Optional[Tensor]:
        """
        Route θ through LGATr's scalar-channel pathway.

        ``mode="channels"`` broadcasts θ[e] to every particle token of event e,
        matching how preprocessed / MET scalars are handled.

        ``mode="tokens"`` lays out ``theta_dim`` dedicated θ tokens per event
        (consistent with ``ptr2index(mode="tokens")``): θ[e, i] is placed at
        slot i of the i-th θ token; all other slots / tokens stay zero. This
        mirrors the transformer's tokens-mode layout, adapted to LGATr's
        scalar pathway.

        Returns a tensor of shape ``(1, n_tokens, theta_dim)``.
        """
        if theta is None:
            return None
        if theta.ndim != 2 or theta.shape[-1] == 0:
            return None

        ptr = ptr.long()
        counts = ptr[1:] - ptr[:-1]
        theta_dim = int(theta.shape[-1])

        if mode == "channels":
            return theta.repeat_interleave(counts, dim=0).unsqueeze(0)

        if mode == "tokens":
            n_events = int(ptr.shape[0] - 1)
            device = theta.device
            tokens_per_event = counts + theta_dim
            tokens_ptr = torch.zeros(n_events + 1, dtype=torch.long, device=device)
            tokens_ptr[1:] = tokens_per_event.cumsum(dim=0)
            n_total = int(tokens_ptr[-1])

            out = theta.new_zeros(n_total, theta_dim)
            theta_offsets = tokens_ptr[:-1]
            dim_idx = torch.arange(theta_dim, device=device)
            theta_rows = theta_offsets.unsqueeze(1) + dim_idx.unsqueeze(0)
            theta_cols = dim_idx.unsqueeze(0).expand(n_events, -1)
            out[theta_rows, theta_cols] = theta
            return out.unsqueeze(0)

        raise ValueError(f"Invalid mode {mode}")

    def forward(
        self,
        particles: Tensor,
        ptr: Tensor,
        scalars: Optional[Tensor] = None,
        met: Optional[Tensor] = None,
        force_math: bool = False,
        embedding_kwargs: Optional[Dict] = None,
    ) -> Tensor:
        """
        Forward for any LGATr wrapper. Let subclasses decide how to embed
        the particles' fourmomenta into multivectors as well as how to
        extract the output from the forwarded tensors.

        :param self: Description
        :param particles: Description
        :type particles: Tensor
        :param ptr: Description
        :type ptr: Tensor
        :param scalars: Description
        :type scalars: Optional[Tensor]
        :param met: Event-level MET features (pt, phi) to add as scalar channels
        :type met: Optional[Tensor]
        :param force_math: Description
        :type force_math: bool
        :param embedding_kwargs: Description
        :type embedding_kwargs: Dict
        :return: Forwarded multivectors and scalars
        :rtype: Tesor
        """
        # If I want to compute the gradient of the ouitput w.r.t. the
        # parameters I cannot use efficient backends for self attention
        backends = get_backends(force_math)

        if embedding_kwargs is None:
            embedding_kwargs = {}
        else:
            embedding_kwargs = dict(embedding_kwargs)

        mode = embedding_kwargs.get("mode", self.mode)
        theta_dim = embedding_kwargs.get("theta_dim", 0)
        theta = embedding_kwargs.get("theta", None)

        # Scalar channels carry, in order: θ | preprocessed | MET. θ is
        # Lorentz-invariant, so it belongs in the scalar pathway rather than
        # the multivector pathway where it would occupy mostly-zero GA slots.
        scalar_parts = []

        theta_scalars = self._expand_theta_scalars(theta, ptr, mode=mode)
        if theta_scalars is not None:
            scalar_parts.append(theta_scalars)

        preprocessed_scalars = self._expand_event_scalars(
            scalars, ptr, mode=mode, theta_dim=theta_dim,
        )
        if preprocessed_scalars is not None and preprocessed_scalars.shape[-1] > 0:
            scalar_parts.append(preprocessed_scalars)

        if met is not None and met.shape[-1] > 0:
            scalar_parts.append(
                self._expand_event_scalars(met, ptr, mode=mode, theta_dim=theta_dim)
            )

        scalars = torch.cat(scalar_parts, dim=-1) if scalar_parts else None

        # Create masking matrix for self-attention
        index = ptr2index(ptr, mode=mode, theta_dim=theta_dim)
        attention_mask = att_mask(index)

        # Embed fourmomenta
        mv = self.embed_mv(particles, **embedding_kwargs)

        # Run forward using only allowed backends
        with sdpa_kernel(backends):
            # out_mv.size() -> (batch_idx, particles, out_mv_channels, 16)
            out_mv, out_s = self.net(
                multivectors=mv, scalars=scalars, attn_mask=attention_mask
            )

        return self.output(multivectors=out_mv, scalars=out_s, index=index)


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
        return to_multivector(particles).repeat(1, 1, theta_dim, 1)

    def output(
        self, multivectors: Tensor, index: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """
        Reduce the multivector batch of each event taking the mean over
        the particles dimension `(dim=1)` and select as ouptut the scalar
        component of the resulting averaged multivectors
        The number of multivector channels is matched with the number of score
        components (see `embed_mv`)

        :param multivectors: Forwarded multivectors
        :type multivectors: Tensor
        :param index: Event index
        :type index: Tensor
        :param scalars: Scalar channels
        :type scalars: Optional[Tensor]
        :return: Forwarded multivectors
        :rtype: Tesor
        """
        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")

        # I assume there are as many output multivector channels as
        # number of dimensions of the score vector to be regressed
        return extract_scalar(out)[0, :, :, 0]  # (batch_size, out_mv_channels)


class ParametrizedLGATrWrapper(BaseLGATrWrapper):
    def embed_mv(
        self,
        particles: Tensor,
        theta: Tensor,
        ptr: Tensor,
        mode: Literal["tokens", "channels"],
        **kwargs,
    ) -> Tensor:
        """
        Embed particle 4-momenta into multivectors. θ is **not** placed in the
        multivector pathway — it is Lorentz-invariant and routed through the
        scalar channels instead (see ``BaseLGATrWrapper.forward``).

        In ``mode="tokens"``, ``theta_dim`` zero-multivector tokens are
        prepended to each event so that the multivector stream has the same
        token layout as the scalar stream (θ tokens then particle tokens).
        """
        mv = to_multivector(particles)  # (1, n_particles, 1, 16)
        if mode == "channels":
            return mv

        if mode == "tokens":
            theta_dim = int(theta.shape[-1])
            if theta_dim == 0:
                return mv

            ptr = ptr.to(dtype=torch.long)
            counts = ptr[1:] - ptr[:-1]
            n_events = int(ptr.shape[0] - 1)
            device = mv.device

            tokens_per_event = counts + theta_dim
            tokens_ptr = torch.zeros(n_events + 1, dtype=torch.long, device=device)
            tokens_ptr[1:] = tokens_per_event.cumsum(dim=0)
            n_total = int(tokens_ptr[-1])

            _, _, c, d = mv.shape
            out = mv.new_zeros(1, n_total, c, d)

            particle_offsets = tokens_ptr[:-1] + theta_dim
            event_of_particle = torch.arange(n_events, device=device).repeat_interleave(counts)
            within_event = (
                torch.arange(int(ptr[-1]), device=device)
                - ptr[:-1].repeat_interleave(counts)
            )
            particle_rows = particle_offsets[event_of_particle] + within_event
            out[0, particle_rows] = mv[0]
            return out

        raise ValueError(f"Invalid mode {mode}")

    def output(
        self, multivectors: Tensor, index: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """
        Reduce the multivector batch of each event by mean-pooling over tokens
        and return the scalar component of the (single) output multivector
        channel as the regressed log-likelihood ratio.
        """
        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")
        return extract_scalar(out)[0, :, :1, 0]  # (batch_size, 1)

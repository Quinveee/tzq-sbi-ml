"""LGATr encoder + conditional normalizing flow wrapper.

The LGATr backbone learns a Lorentz-aware per-token representation of each
event. Tokens are pooled into a single event-level vector ``z`` using a
learned attention weighting. A conditional normalizing flow then models
``p(z | theta)``; the negative log-likelihood is the training loss, and
log-likelihood ratios / scores are recovered at evaluation time from
``log p(z | theta_1) - log p(z | theta_0)``.

Note: the reference sketch had ``context = cat([z, theta])``, which is
degenerate — the flow can trivially memorize z through the context. We use
``context = theta`` (plus optional extra scalars) instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .embed import to_multivector
from .utils import att_mask, get_backends, ptr2index

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import Tensor


class LGATrCNFWrapper(BaseWrapper):
    """LGATr encoder + learned-attention pooling + conditional RealNVP."""

    def __init__(
        self,
        net: "DictConfig",
        flow: "DictConfig",
        token_dim: int,
        pool_dim: int,
        theta_dim: int,
        **_,
    ) -> None:
        super().__init__(net=None, key="LGATrCNF")
        self.net = instantiate(net)
        self.flow = instantiate(
            flow, dim=pool_dim, context_dim=theta_dim
        )

        # Learned attention over tokens -> single weight per token
        self.pool_attn = nn.Linear(token_dim, 1)
        # Projection from LGATr per-token features to the flow's latent dim
        self.pool_proj = nn.Linear(token_dim, pool_dim)

        nn.init.xavier_uniform_(self.pool_attn.weight)
        nn.init.zeros_(self.pool_attn.bias)
        nn.init.xavier_uniform_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)

        self._token_dim = token_dim
        self._pool_dim = pool_dim

    @staticmethod
    def _expand_event_scalars(scalars: "Tensor", ptr: "Tensor") -> "Tensor":
        """Repeat event-level scalars to token-level: (B, s) -> (1, N_tokens, s)."""
        ptr = ptr.long()
        repeats = ptr[1:] - ptr[:-1]
        expanded = scalars.repeat_interleave(repeats, dim=0)  # (N_tokens, s)
        return expanded.unsqueeze(0)

    @staticmethod
    def _masked_softmax(scores: "Tensor", index: "Tensor", num_events: int) -> "Tensor":
        """Per-event softmax over flat tokens identified by ``index``."""
        max_per_event = scatter(
            scores, index=index, dim=0, dim_size=num_events, reduce="max"
        )
        shifted = scores - max_per_event[index]
        exp = shifted.exp()
        denom = scatter(
            exp, index=index, dim=0, dim_size=num_events, reduce="sum"
        ).clamp_min(1e-12)
        return exp / denom[index]

    def forward(
        self,
        particles: "Tensor",
        theta: "Tensor",
        ptr: "Tensor",
        scalars: Optional["Tensor"] = None,
        force_math: bool = False,
    ) -> "Tensor":
        """Returns per-event log p(z | theta) of shape (B,)."""
        # Embed particle 4-momenta as multivectors. Theta is NOT mixed into the
        # encoder; it conditions only the flow.
        mv = to_multivector(particles)  # (1, N_tokens, 1, 16)

        index = ptr2index(ptr, mode="channels")  # (N_tokens,) with event id
        attention_mask = att_mask(index)

        # Expand event-level scalars (e.g. MET pt/phi) to token-level.
        if scalars is not None and scalars.ndim == 2 and scalars.shape[-1] > 0:
            scalars = self._expand_event_scalars(scalars, ptr)
        elif scalars is not None and scalars.shape[-1] == 0:
            scalars = None

        backends = get_backends(force_math)
        with sdpa_kernel(backends):
            out_mv, out_s = self.net(
                multivectors=mv, scalars=scalars, attn_mask=attention_mask
            )

        # Flatten per-token features: concat all 16 multivector components per
        # channel with scalar channels -> token_dim
        h = out_mv.squeeze(0).flatten(-2)  # (N_tokens, out_mv_channels * 16)
        if out_s is not None:
            h = torch.cat([h, out_s.squeeze(0)], dim=-1)

        assert h.shape[-1] == self._token_dim, (
            f"Configured token_dim={self._token_dim} but LGATr produced {h.shape[-1]}. "
            "Set model.token_dim = out_mv_channels*16 + out_s_channels."
        )

        # Learned attention pooling per event
        num_events = theta.shape[0]
        logits = self.pool_attn(h).squeeze(-1)  # (N_tokens,)
        weights = self._masked_softmax(logits, index, num_events)  # (N_tokens,)

        h_proj = self.pool_proj(h)  # (N_tokens, pool_dim)
        z = scatter(
            weights.unsqueeze(-1) * h_proj,
            index=index,
            dim=0,
            dim_size=num_events,
            reduce="sum",
        )  # (B, pool_dim)

        return self.flow.log_prob(z, context=theta)  # (B,)

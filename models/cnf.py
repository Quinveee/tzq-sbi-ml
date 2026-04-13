"""Conditional normalizing flow (affine coupling, Real-NVP style).

Minimal self-contained implementation so we don't pull in a new dependency.
The flow models p(x | context); the caller supplies `context` as a tensor
concatenating whatever features they want the flow to condition on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConditionerMLP(nn.Module):
    """Context-aware MLP that predicts (log-scale, shift) for a coupling layer."""

    def __init__(self, in_dim: int, context_dim: int, hidden_dim: int, n_hidden: int, out_dim: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim + context_dim, hidden_dim), nn.GELU()]
        for _ in range(max(n_hidden - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        last = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x_masked: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_masked, context], dim=-1))


class CondAffineCoupling(nn.Module):
    """Conditional affine coupling layer with alternating binary masks."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        mask_type: str = "even",
    ):
        super().__init__()
        mask = torch.zeros(dim)
        if mask_type == "even":
            mask[::2] = 1.0
        elif mask_type == "odd":
            mask[1::2] = 1.0
        else:
            raise ValueError(f"Invalid mask_type {mask_type}")
        self.register_buffer("mask", mask)
        self.conditioner = _ConditionerMLP(
            in_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            out_dim=2 * dim,
        )

    def _params(self, x: torch.Tensor, context: torch.Tensor):
        h = self.conditioner(x * self.mask, context)
        s, t = h.chunk(2, dim=-1)
        # tanh bounds the log-scale for stability; zero out transformed half
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        s, t = self._params(x, context)
        y = x * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        s, t = self._params(y, context)  # mask ensures conditioner sees y_masked == x_masked
        x = (y - t) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        return x, log_det


class ConditionalRealNVP(nn.Module):
    """Stack of conditional affine coupling layers over a standard-normal base."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_flows: int = 8,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList(
            [
                CondAffineCoupling(
                    dim=dim,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    n_hidden=n_hidden,
                    mask_type="even" if i % 2 == 0 else "odd",
                )
                for i in range(n_flows)
            ]
        )

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        y = x
        for f in self.flows:
            y, log_det = f(y, context)
            log_det_total = log_det_total + log_det
        log_base = -0.5 * (y**2).sum(dim=-1) - 0.5 * self.dim * math.log(2.0 * math.pi)
        return log_base + log_det_total

    @torch.no_grad()
    def sample(self, n_samples: int, context: torch.Tensor) -> torch.Tensor:
        z = torch.randn(n_samples, self.dim, device=context.device, dtype=context.dtype)
        for f in reversed(self.flows):
            z, _ = f.inverse(z, context)
        return z

"""LorentzNet architecture

Adapted from https://github.com/sdogsq/LorentzNet-release/blob/main/models.py.
Operates on padded particle batches of shape ``(B, L_max, ...)`` with a
``valid_mask`` of shape ``(B, L_max)`` indicating real particles. Message
passing is the fully-connected (per event, no self-loops) variant of the
original model, vectorized as dense ``(B, L, L)`` pair tensors and masked.
"""

from __future__ import annotations

import torch
from torch import nn


def normsq4(p: torch.Tensor) -> torch.Tensor:
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


def _bn_on_last(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """Apply a BatchNorm1d (built for (N, C)) to a (..., C) tensor by
    folding leading dims, normalizing, then unfolding."""
    *leading, C = x.shape
    return bn(x.reshape(-1, C)).reshape(*leading, C)


class LGEB(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 0,
        dropout: float = 0.0,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ):
        super().__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # Minkowski norm and inner product

        # phi_e has a BatchNorm in the middle; apply it manually so it works
        # on (B, L, L, C) tensors.
        self.phi_e_lin1 = nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False)
        self.phi_e_bn = nn.BatchNorm1d(n_hidden)
        self.phi_e_lin2 = nn.Linear(n_hidden, n_hidden)

        self.phi_h_lin1 = nn.Linear(n_hidden + n_input + n_node_attr, n_hidden)
        self.phi_h_bn = nn.BatchNorm1d(n_hidden)
        self.phi_h_lin2 = nn.Linear(n_hidden, n_output)

        layer = nn.Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer,
        )

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def _phi_e(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, L, n_input*2 + n_edge_attr) -> (B, L, L, n_hidden)
        h = self.phi_e_lin1(x)
        h = _bn_on_last(self.phi_e_bn, h)
        h = torch.relu(h)
        h = self.phi_e_lin2(h)
        return torch.relu(h)

    def _phi_h(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, n_hidden + n_input + n_node_attr) -> (B, L, n_output)
        h = self.phi_h_lin1(x)
        h = _bn_on_last(self.phi_h_bn, h)
        h = torch.relu(h)
        return self.phi_h_lin2(h)

    def forward(
        self,
        h: torch.Tensor,           # (B, L, n_input)
        x: torch.Tensor,           # (B, L, 4)
        node_attr: torch.Tensor,   # (B, L, n_node_attr)
        edge_mask: torch.Tensor,   # (B, L, L) bool — valid pairs (no self, no pad)
        node_mask: torch.Tensor,   # (B, L) bool — valid nodes
    ):
        B, L, _ = h.shape

        # Pairwise Minkowski features. x_diff[b, i, j] = x[b, i] - x[b, j].
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)             # (B, L, L, 4)
        norms = psi(normsq4(x_diff)).unsqueeze(-1)           # (B, L, L, 1)
        dots = psi(dotsq4(x.unsqueeze(2), x.unsqueeze(1))).unsqueeze(-1)

        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)           # (B, L, L, n_input)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)
        m_in = torch.cat([h_i, h_j, norms, dots], dim=-1)
        m = self._phi_e(m_in)                                # (B, L, L, n_hidden)
        m = m * self.phi_m(m)                                # gate
        # Zero out invalid pairs so they don't contribute to any aggregate.
        m = m * edge_mask.unsqueeze(-1).to(m.dtype)

        # Coordinate update (skipped on the last layer).
        if not self.last_layer:
            trans = x_diff * self.phi_x(m)                   # (B, L, L, 4)
            trans = trans * edge_mask.unsqueeze(-1).to(trans.dtype)
            trans = torch.clamp(trans, min=-100, max=100)
            # Per-receiver mean over the j axis (dim 2), restricted to valid neighbors.
            denom = edge_mask.sum(dim=2, keepdim=True).clamp(min=1).unsqueeze(-1).to(trans.dtype)
            agg = trans.sum(dim=2) / denom.squeeze(-1)       # (B, L, 4)
            x = x + agg * self.c_weight

        # Node feature update: aggregate messages by sum over j.
        agg_m = m.sum(dim=2)                                 # (B, L, n_hidden)
        h_in = torch.cat([h, agg_m, node_attr], dim=-1)
        h = h + self._phi_h(h_in)

        # Zero pad-rows so downstream layers don't propagate garbage statistics.
        node_mask_f = node_mask.unsqueeze(-1).to(h.dtype)
        h = h * node_mask_f
        x = x * node_mask_f
        return h, x, m


class LorentzNet(nn.Module):
    r"""LorentzNet operating on padded particle batches.

    :param n_scalar: Dimension of per-particle scalar input features.
    :param n_hidden: Latent hidden dimension.
    :param dim_out: Output dimension (per event).
    :param n_layers: Number of LGEB layers.
    :param c_weight: Coordinate update scaling factor.
    :param dropout: Dropout rate in the graph decoder head.
    """

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        dim_out: int,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    n_hidden,
                    n_hidden,
                    n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, dim_out),
        )

    def forward(
        self,
        scalars: torch.Tensor,    # (B, L, n_scalar)
        x: torch.Tensor,          # (B, L, 4)
        valid_mask: torch.Tensor, # (B, L) bool
    ) -> torch.Tensor:
        B, L = valid_mask.shape
        eye = torch.eye(L, dtype=torch.bool, device=valid_mask.device)
        edge_mask = (
            valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & ~eye
        )  # (B, L, L)

        h = self.embedding(scalars)
        for layer in self.LGEBs:
            h, x, _ = layer(
                h, x, node_attr=scalars,
                edge_mask=edge_mask, node_mask=valid_mask,
            )

        # Masked mean over real particles -> (B, n_hidden)
        weight = valid_mask.unsqueeze(-1).to(h.dtype)
        counts = weight.sum(dim=1).clamp(min=1)
        pooled = (h * weight).sum(dim=1) / counts
        return self.graph_dec(pooled)

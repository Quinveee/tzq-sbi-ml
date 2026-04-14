"""LorentzNet architecture

Adapted from https://github.com/sdogsq/LorentzNet-release/blob/main/models.py
to operate on flattened particle tensors with an event pointer, matching the
conventions used for the Transformer and LGATr wrappers in this project.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter


def normsq4(p: torch.Tensor) -> torch.Tensor:
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


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

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

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

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        return out * w

    def h_model(self, h, edges, m, node_attr):
        i, _ = edges
        agg = scatter(m, i, dim=0, dim_size=h.size(0), reduce="sum")
        agg = torch.cat([h, agg, node_attr], dim=1)
        return h + self.phi_h(agg)

    def x_model(self, x, edges, x_diff, m):
        i, _ = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = scatter(trans, i, dim=0, dim_size=x.size(0), reduce="mean")
        return x + agg * self.c_weight

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        return psi(norms), psi(dots), x_diff

    def forward(self, h, x, edges, node_attr):
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.m_model(h[edges[0]], h[edges[1]], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


def build_fully_connected_edges(ptr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fully-connected (no self-loops) edge indices per event in a batch
    of flattened particles described by `ptr`.
    """
    ptr_long = ptr.to(dtype=torch.long)
    lengths = ptr_long[1:] - ptr_long[:-1]
    device = ptr_long.device

    src_list, dst_list = [], []
    for start, n in zip(ptr_long[:-1].tolist(), lengths.tolist()):
        if n <= 1:
            continue
        idx = torch.arange(start, start + n, device=device)
        i = idx.repeat_interleave(n)
        j = idx.repeat(n)
        mask = i != j
        src_list.append(i[mask])
        dst_list.append(j[mask])

    if not src_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(src_list), torch.cat(dst_list)


class LorentzNet(nn.Module):
    r"""LorentzNet for flattened particle batches.

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
        scalars: torch.Tensor,
        x: torch.Tensor,
        ptr: torch.Tensor,
    ) -> torch.Tensor:
        edges = build_fully_connected_edges(ptr)
        h = self.embedding(scalars)
        for layer in self.LGEBs:
            h, x, _ = layer(h, x, edges, node_attr=scalars)

        ptr_long = ptr.to(dtype=torch.long)
        index = torch.arange(
            ptr_long.numel() - 1, device=h.device
        ).repeat_interleave(ptr_long[1:] - ptr_long[:-1])
        pooled = scatter(h, index, dim=0, reduce="mean")
        return self.graph_dec(pooled)

"""Structured linear input projection for the Transformer.

Splits an input token of shape ``(..., n_scalar_in + 4*n_vec_in)`` into a
Lorentz-scalar branch and a Lorentz-vector branch, projects each to a disjoint
subset of the per-head embedding so that, for every head, the layout is

    [ leading scalars (n_head_scalars)
    | vectors (4 * n_head_vectors)
    | trailing scalars ]

Scalars-from-vectors and vectors-from-scalars mixing are forbidden. The vector
branch is an equivariant scalar combination of input 4-vectors (no bias and no
per-component mixing), so output channels marked as "vector" actually transform
as 4-vectors under the LLoCa attention's frame transport. The non-LLoCa
transformer uses the same layer so the parameter layout is shared.

Token layout convention: scalars come first, the trailing ``4 * n_vec_in``
channels carry the input 4-vectors stacked in (channel, spacetime) order. The
existing transformer wrappers concatenate ``[theta | preprocessed | met |
particle 4-momentum]``, so this lines up with their output without changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StructuredLinearIn(nn.Module):
    def __init__(
        self,
        n_scalar_in: int,
        n_vec_in: int,
        emb_hidden: int,
        num_heads: int,
        n_head_scalars: int,
        n_head_vectors: int,
    ) -> None:
        super().__init__()
        if emb_hidden % num_heads != 0:
            raise ValueError(
                f"emb_hidden={emb_hidden} must be divisible by num_heads={num_heads}"
            )
        head_dim = emb_hidden // num_heads
        # No vector inputs -> no point reserving vector slots in each head.
        if n_vec_in == 0:
            n_head_vectors = 0
        d_vec = n_head_vectors * 4
        if n_head_scalars + d_vec > head_dim:
            raise ValueError(
                f"head_dim={head_dim} cannot fit n_head_scalars={n_head_scalars} "
                f"+ 4*n_head_vectors={d_vec}"
            )

        self.n_scalar_in = int(n_scalar_in)
        self.n_vec_in = int(n_vec_in)
        self.emb_hidden = int(emb_hidden)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.n_head_scalars = int(n_head_scalars)
        self.n_head_vectors = int(n_head_vectors)
        self.d_vec = int(d_vec)
        self.n_scalar_slots = int(head_dim - d_vec)  # leading + trailing per head

        # Scalar branch: bias allowed.
        if self.n_scalar_in > 0 and self.n_scalar_slots > 0:
            self.scalar_proj: nn.Linear | None = nn.Linear(
                self.n_scalar_in, self.num_heads * self.n_scalar_slots
            )
        else:
            self.scalar_proj = None

        # Vector branch: equivariant scalar combination across input 4-vectors.
        if self.n_vec_in > 0 and self.n_head_vectors > 0:
            self.vec_weight = nn.Parameter(
                torch.empty(self.num_heads * self.n_head_vectors, self.n_vec_in)
            )
            nn.init.xavier_uniform_(self.vec_weight)
        else:
            self.register_parameter("vec_weight", None)

    @property
    def in_features(self) -> int:
        # Compat shim for code that introspects ``linear_in.in_features``.
        return self.n_scalar_in + 4 * self.n_vec_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_dims = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]

        scalars = x_flat[:, : self.n_scalar_in]
        if self.n_vec_in > 0:
            v_dim = 4 * self.n_vec_in
            vecs = x_flat[:, self.n_scalar_in : self.n_scalar_in + v_dim].reshape(
                N, self.n_vec_in, 4
            )
        else:
            vecs = None

        if self.scalar_proj is not None:
            scalar_emb = self.scalar_proj(scalars).reshape(
                N, self.num_heads, self.n_scalar_slots
            )
        else:
            scalar_emb = x_flat.new_zeros(N, self.num_heads, self.n_scalar_slots)

        leading = scalar_emb[..., : self.n_head_scalars]
        trailing = scalar_emb[..., self.n_head_scalars :]

        if self.vec_weight is not None and vecs is not None:
            # (N, V_in, 4) x (V_out, V_in) -> (N, V_out, 4) where V_out = H * n_head_vectors
            vec_out = torch.einsum("nvc,ov->noc", vecs, self.vec_weight)
            vec_out = vec_out.reshape(N, self.num_heads, self.d_vec)
        else:
            vec_out = x_flat.new_zeros(N, self.num_heads, self.d_vec)

        out = torch.cat([leading, vec_out, trailing], dim=-1)  # (N, H, head_dim)
        return out.reshape(*leading_dims, self.emb_hidden)

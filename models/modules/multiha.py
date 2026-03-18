"""Multi-head attention module"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..configs import SAConfig


class MultiHA(nn.Module):
    def __init__(self, config: SAConfig):
        super().__init__()

        self.config = config
        self.packed_proj = nn.Linear(config.emb_size, config.emb_size * 3, bias=config.bias)
        self.unify_heads = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_p) if config.dropout_p else None

    def forward(self, x: torch.Tensor, lloca: bool = False, attn_mask: torch.Tensor | None = None, **kwargs):
        """
        x: (batch*particles, emb_size) -> emb_size obtained from embedding layer which converted 
            the input four-momenta + wilson coefficients into the embedding dimension
        attn_mask: optional attention mask of shape (batch*particles, batch*particles)
        lloca: whether to use lorentz equivariant attention for lloca or standard multi-head attention
        """
        b, e = x.size()

        print(f"Input to MultiHA: {x.size()}")

        assert e == self.config.emb_size, f"Embedding size mismatch: {e} != {self.config.emb_size}"

        result = self.packed_proj(x)
        query, key, value = torch.chunk(result, 3, dim=-1)

        query = (
            query.unflatten(-1, (self.config.num_heads, self.config.emb_head))
            .transpose(0, 1)
            .contiguous()
        )

        key = (
            key.unflatten(-1, (self.config.num_heads, self.config.emb_head))
            .transpose(0, 1)
            .contiguous()
        )

        value = (
            value.unflatten(-1, (self.config.num_heads, self.config.emb_head))
            .transpose(0, 1)
            .contiguous()
        )

        # confirm shapes: (N_heads, batch*particles, emb_head)
        assert (
            query.size()
            == key.size()
            == value.size()
            == (self.config.num_heads, b, self.config.emb_head)
        )

        # Ensure attn_mask has correct dtype and shape for cuBLAS kernels
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # broadcast to heads
            elif attn_mask.dim() != 3:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.shape}")

        if lloca:            
            pass

        else: 
            # Scaled dot-product attention with optional attn mask and dropout
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.config.dropout_p if self.training else 0.0,
                **kwargs
            )

        # Merge heads: (N_heads, batch*particles, emb_head) -> (batch*particles, emb_size)
        out = out.transpose(0, 1).flatten(-2)
        assert out.size() == (b, self.config.emb_size)

        # Final linear projection and dropout
        out = self.unify_heads(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out
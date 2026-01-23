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

        self.packed_proj = nn.Linear(
            config.emb_size, config.emb_size * 3, bias=config.bias
        )
        self.unify_heads = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)
        self.dropout = (
            nn.Dropout(config.dropout_p) if config.dropout_p is not None else None
        )
        self.config = config

    def forward(self, x: torch.Tensor, **attn_kwds):
        """
        We flatten batch and sequence length dimensions and use an attn matrix
        to separate particles belonging to different events
        """
        b, e = x.size()

        assert (
            e == self.config.emb_size
        ), f"Embedding size doesn't match: found: {e}, expected: {self.config.emb_size}"

        result = self.packed_proj(x)
        query, key, value = torch.chunk(result, 3, dim=-1)

        query = query.unflatten(
            -1, [self.config.num_heads, self.config.emb_head]
        ).transpose(0, 1)
        key = key.unflatten(
            -1, [self.config.num_heads, self.config.emb_head]
        ).transpose(0, 1)
        value = value.unflatten(
            -1, [self.config.num_heads, self.config.emb_head]
        ).transpose(0, 1)

        assert (
            query.size()
            == key.size()
            == value.size()
            == (self.config.num_heads, b, self.config.emb_head)
        )

        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.config.dropout_p, **attn_kwds
        )

        out = out.transpose(0, 1).flatten(-2)

        assert out.size() == (b, self.config.emb_size)

        out = self.unify_heads(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out

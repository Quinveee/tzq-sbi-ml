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
        Multi-head attention forward pass, safe for 2D (batch, emb_size) input.
        If input has no sequence dimension, we add one. Our framework assumes
        the input to be (batch, emb_size).
        """

        # If input is 2D, add a fake sequence dimension
        if x.dim() == 2:
            # x: (batch, emb_size) -> (batch, seq_len=1, emb_size)
            x = x.unsqueeze(1)

        print(x.size())
        b, s, e = x.size()  # batch, seq_len, emb_size

        assert (
            e == self.config.emb_size
        ), f"Embedding size doesn't match: found {e}, expected {self.config.emb_size}"

        # Linear projections for Q, K, V
        result = self.packed_proj(x)  # (batch, seq_len, 3*emb_size)
        query, key, value = torch.chunk(result, 3, dim=-1)  # each: (batch, seq_len, emb_size)

        # Split heads: (batch, seq_len, emb_size) -> (batch, seq_len, num_heads, head_dim)
        H = self.config.num_heads
        D = self.config.emb_head  # emb_size // num_heads
        query = query.view(b, s, H, D).transpose(1, 2).contiguous()  # (batch, heads, seq_len, head_dim)
        key   = key.view(b, s, H, D).transpose(1, 2).contiguous()
        value = value.view(b, s, H, D).transpose(1, 2).contiguous()

        # Attention: (batch, heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.config.dropout_p)

        # This implementation uses attention masking which does not make sense to use when the sequence length is 1:
        # out = F.scaled_dot_product_attention(
        #     query, key, value, dropout_p=self.config.dropout_p, **attn_kwds
        # )  # same shape

        # Merge heads: (batch, heads, seq_len, head_dim) -> (batch, seq_len, emb_size)
        out = out.transpose(1, 2).reshape(b, s, e)

        # If we added a fake sequence dimension, remove it
        if s == 1:
            out = out.squeeze(1)  # -> (batch, emb_size)

        # Final linear
        out = self.unify_heads(out)

        return out

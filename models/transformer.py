"""Transformer architecture"""

from dataclasses import replace
from typing import Mapping

import torch.nn as nn

from .configs import MLPConfig, SAConfig
from .modules.te import TE


def derive_emb_hidden(dim_in: int, emb_factor: int, num_heads: int) -> int:
    """
    Derives the maximum allowed embedding dimension for hidden states
    based on the increasing factor and the number of attention heads requested

    :param dim_in: Feature dimension
    :type dim_in: int
    :param emb_factor: Increasing factor
    :type emb_factor: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :return: Hidden embedding dimension
    :rtype: int
    """
    emb_candidate = dim_in * emb_factor
    emb = emb_candidate - (emb_candidate % num_heads)
    return max(emb, num_heads)


class Transformer(nn.Module):
    """
    Transformer architecture
    """

    def __init__(
    self,
    dim_in: int,
    emb_factor: int,
    dim_out: int,
    num_blocks: int,
    attention: Mapping,
    mlp: Mapping,
    dropout_p: float,
    lloca_num_scalars: int = 0,   # ← add
    lloca_num_vectors: int = 0,   # ← add
    ) -> None:
        super().__init__()
        emb_hidden = derive_emb_hidden(dim_in, emb_factor, attention.num_heads)

        if lloca_num_vectors:
            emb_head = emb_hidden // attention.num_heads
            min_head = lloca_num_scalars + lloca_num_vectors * 4
            assert emb_head >= min_head, (
                f"LLoCa requires emb_head ({emb_head}) ≥ "
                f"n_scalars + n_vectorsx4 = {min_head}. "
                f"Increase emb_factor or reduce LLoCa vector channels."
            )

        # configs
        attention = replace(
            SAConfig.cast(attention), emb_size=emb_hidden, dropout_p=dropout_p
        )
        mlp = replace(
            MLPConfig.cast(mlp),
            dim_in=emb_hidden,
            dim_out=emb_hidden,
            dropout_p=dropout_p,
        )

        # layers
        self.linear_in = nn.Linear(dim_in, emb_hidden)
        self.te_blocks = nn.ModuleList(
            [
                TE(
                    emb_size=emb_hidden,
                    attention=attention,
                    mlp=mlp,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(emb_hidden, dim_out)

    def forward(self, x, **attn_kwds):
        x = self.linear_in(x)
        for layer in self.te_blocks:
            x = layer(x, **attn_kwds)
        return self.linear_out(x)

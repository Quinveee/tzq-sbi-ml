"""Transformer architecture"""

import math
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
    lloca_num_scalars: int = 0,
    lloca_num_vectors: int = 0,
    ) -> None:
        super().__init__()
        emb_hidden = derive_emb_hidden(dim_in, emb_factor, attention.num_heads)

        if lloca_num_vectors:
            emb_head = emb_hidden // attention.num_heads
            min_head = lloca_num_scalars + lloca_num_vectors * 4
            if emb_head < min_head:
                raise ValueError(
                    "Invalid LLoCa attention setup: "
                    f"emb_head={emb_head}, required={min_head} "
                    f"(n_scalars={lloca_num_scalars}, n_vectors={lloca_num_vectors}). "
                    "Increase emb_factor / embedding size, reduce num_heads, "
                    "or reduce LLoCa scalar/vector channels."
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

        self._init_weights(num_blocks)

    def _init_weights(self, num_blocks: int) -> None:
        # Xavier on input/output projections, Kaiming on MLP hidden layer
        # (GELU ~ ReLU regime), and rescale the residual-output projections
        # (attention `unify_heads` and the MLP's last linear) by
        # 1/sqrt(2*num_blocks). This keeps the variance of the residual
        # stream from blowing up with depth in pre-LN transformers.
        nn.init.xavier_uniform_(self.linear_in.weight)
        if self.linear_in.bias is not None:
            nn.init.zeros_(self.linear_in.bias)
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

        residual_gain = 1.0 / math.sqrt(2.0 * max(num_blocks, 1))

        for block in self.te_blocks:
            # Self-attention QKV projection
            nn.init.xavier_uniform_(block.sa.packed_proj.weight)
            if block.sa.packed_proj.bias is not None:
                nn.init.zeros_(block.sa.packed_proj.bias)
            # Self-attention output projection sits on the residual path
            nn.init.xavier_uniform_(block.sa.unify_heads.weight, gain=residual_gain)
            if block.sa.unify_heads.bias is not None:
                nn.init.zeros_(block.sa.unify_heads.bias)

            # MLP block: hidden linears use Kaiming (relu/gelu); the last
            # linear sits on the residual path and gets the depth scaling.
            mlp_linears = [m for m in block.mlp.net if isinstance(m, nn.Linear)]
            for lin in mlp_linears[:-1]:
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            nn.init.xavier_uniform_(mlp_linears[-1].weight, gain=residual_gain)
            if mlp_linears[-1].bias is not None:
                nn.init.zeros_(mlp_linears[-1].bias)

    def forward(self, x, **attn_kwds):
        x = self.linear_in(x)
        for layer in self.te_blocks:
            x = layer(x, **attn_kwds)
        return self.linear_out(x)

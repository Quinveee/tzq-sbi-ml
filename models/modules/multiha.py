"""Multi-head attention module"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import lloca_dot_product_attention   # ← new import

if TYPE_CHECKING:
    from ..configs import SAConfig


class MultiHA(nn.Module):
    def __init__(self, config: SAConfig):
        super().__init__()
        self.config = config
        self.packed_proj = nn.Linear(config.emb_size, config.emb_size * 3, bias=config.bias)
        self.unify_heads = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_p) if config.dropout_p else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_kwargs: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        b, e = x.size()
        assert e == self.config.emb_size

        # ── Unpack LLoCa config ──────────────────────────────────────────────
        lloca = False
        lloca_frames = lloca_frames_inv = None
        lloca_num_scalars = lloca_num_vectors = 0
        frames = frames_inv = None

        if attn_kwargs:
            lloca = attn_kwargs.get("lloca", False)
            lloca_num_scalars = attn_kwargs.get("lloca_num_scalars", 0)
            lloca_num_vectors = attn_kwargs.get("lloca_num_vectors", 0)
            frames = attn_kwargs.get("frames", None)
            frames_inv = attn_kwargs.get("frames_inv", None)

        b, e = x.size()

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
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() != 3:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.shape}")

        # Forward pass
        if lloca:
            if frames is None:
                raise ValueError(
                    "LLoCa is active but 'frames' was not found in attn_kwargs. "
                    "Make sure build_lloca_frames() is called in the wrapper and "
                    "stored under attn_kwargs['frames']."
                )
            # Primary attention using K frames
            out = lloca_dot_product_attention(
                query, key, value,
                frames=frames,
                n_scalars=lloca_num_scalars,
                n_vectors=lloca_num_vectors,
                attn_mask=attn_mask,
                dropout_p=self.config.dropout_p if self.training else 0.0,
                training=self.training
            )
            # Optional: secondary invariant-frame contribution
            # (frames_inv provides an alternative neighbourhood basis;
            #  add its contribution with its own scalar/vector split if desired)
            if frames_inv is not None and frames_inv is not frames:
                out = out + lloca_dot_product_attention(
                    query, key, value,
                    frames=frames_inv,
                    n_scalars=lloca_num_scalars,
                    n_vectors=lloca_num_vectors,
                    attn_mask=attn_mask,
                    dropout_p=self.config.dropout_p if self.training else 0.0,
                    training=self.training,
                )
                out = out * 0.5   # average the two contributions
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
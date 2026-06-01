"""Multi-head attention module"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import lloca_dot_product_attention

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
        lloca = False
        lloca_num_scalars = lloca_num_vectors = 0
        frames = inv_frames = None

        if attn_kwargs:
            lloca = attn_kwargs.get("lloca", False)
            lloca_num_scalars = attn_kwargs.get("lloca_num_scalars", 0)
            lloca_num_vectors = attn_kwargs.get("lloca_num_vectors", 0)
            frames = attn_kwargs.get("frames", None)
            inv_frames = attn_kwargs.get("inv_frames", None)

        # Branch on input rank so we can run batched SDPA on padded (B, L, E)
        # inputs without quadratic-in-batch attention masks. LLoCa is handled
        # too: q/k/v are transported through per-particle frames before SDPA.
        if x.dim() == 3:
            return self._forward_padded(
                x,
                attn_mask,
                lloca=lloca,
                lloca_num_scalars=lloca_num_scalars,
                lloca_num_vectors=lloca_num_vectors,
                frames=frames,
                inv_frames=inv_frames,
            )

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

            min_head = lloca_num_scalars + lloca_num_vectors * 4
            if self.config.emb_head < min_head:
                raise ValueError(
                    "Invalid LLoCa attention setup: "
                    f"emb_head={self.config.emb_head}, required={min_head} "
                    f"(n_scalars={lloca_num_scalars}, n_vectors={lloca_num_vectors}, "
                    f"emb_size={self.config.emb_size}, num_heads={self.config.num_heads}). "
                    "Increase emb_factor / embedding size, reduce num_heads, "
                    "or reduce LLoCa scalar/vector channels."
                )

            # Primary attention using K frames
            out = lloca_dot_product_attention(
                query, key, value,
                frames=frames,
                n_scalars=lloca_num_scalars,
                n_vectors=lloca_num_vectors,
                attn_mask=attn_mask,
                inv_frames=inv_frames,
                dropout_p=self.config.dropout_p if self.training else 0.0,
                training=self.training,
            )
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

    def _forward_padded(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        lloca: bool = False,
        lloca_num_scalars: int = 0,
        lloca_num_vectors: int = 0,
        frames=None,
        inv_frames=None,
    ) -> torch.Tensor:
        """
        Padded attention path. Expects:
            x: (B, L, emb_size)
            valid_mask: (B, L) bool — True for real tokens, False for padding.
        When ``lloca`` is True, ``frames`` must carry the per-particle Lorentz
        frames of shape (B, L, 4, 4); LLoCa attention transports q/k/v through
        them before the SDPA call.
        """
        B, L, E = x.size()
        assert E == self.config.emb_size, (
            f"Embedding size mismatch: {E} != {self.config.emb_size}"
        )
        H = self.config.num_heads
        D = self.config.emb_head

        qkv = self.packed_proj(x)  # (B, L, 3E)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # (B, L, H, D) -> (B, H, L, D)
        q = q.unflatten(-1, (H, D)).transpose(1, 2).contiguous()
        k = k.unflatten(-1, (H, D)).transpose(1, 2).contiguous()
        v = v.unflatten(-1, (H, D)).transpose(1, 2).contiguous()

        # Mask out padding *keys* so queries don't pull garbage from padded
        # positions. SDPA bool mask convention: True means "attend". Shape
        # (B, 1, 1, L) broadcasts across heads and queries.
        attn_mask = None
        if valid_mask is not None:
            attn_mask = valid_mask.to(torch.bool)[:, None, None, :]

        if lloca:
            if frames is None:
                raise ValueError(
                    "LLoCa is active but 'frames' was not found. "
                    "Make sure build_lloca_frames() is called in the wrapper."
                )
            min_head = lloca_num_scalars + lloca_num_vectors * 4
            if self.config.emb_head < min_head:
                raise ValueError(
                    "Invalid LLoCa attention setup: "
                    f"emb_head={self.config.emb_head}, required={min_head} "
                    f"(n_scalars={lloca_num_scalars}, n_vectors={lloca_num_vectors})."
                )
            out = lloca_dot_product_attention(
                q,
                k,
                v,
                frames=frames,
                n_scalars=lloca_num_scalars,
                n_vectors=lloca_num_vectors,
                attn_mask=attn_mask,
                inv_frames=inv_frames,
                dropout_p=self.config.dropout_p if self.training else 0.0,
                training=self.training,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.config.dropout_p if self.training else 0.0,
            )

        # (B, H, L, D) -> (B, L, E)
        out = out.transpose(1, 2).contiguous().flatten(-2)
        out = self.unify_heads(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
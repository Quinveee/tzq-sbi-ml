"""
Functions to embed particle four momenta into the GA multivectors
"""

import torch
from lgatr import embed_vector


def to_multivector(fourmomenta: torch.Tensor) -> torch.Tensor:
    """
    Embed four-momenta into multivectors of shape ``(..., items, 1, 16)``.

    Accepts either flat ``(N, 4)`` (returns ``(1, N, 1, 16)`` for the legacy
    flat path) or padded ``(B, L, 4)`` (returns ``(B, L, 1, 16)``).
    """
    if fourmomenta.ndim == 2:
        # Flat (N, 4) -> (1, N, 1, 16). Used by the legacy flat path.
        fourmomenta = fourmomenta.unsqueeze(-2)  # (N, 1, 4)
        mv = embed_vector(fourmomenta)  # (N, 1, 16)
        return mv.unsqueeze(0)
    if fourmomenta.ndim == 3:
        # Padded (B, L, 4) -> (B, L, 1, 16).
        return embed_vector(fourmomenta.unsqueeze(-2))
    raise ValueError(
        f"to_multivector expects fourmomenta of rank 2 or 3, got {fourmomenta.shape}"
    )

"""
Functions to embed particle four momenta into the GA multivectors
"""

import torch
from lgatr import embed_vector


def to_multivector(fourmomenta: torch.Tensor) -> torch.Tensor:
    """
    Embed fourmomenta into multivector and add extra dimensions
    so that output is (batch dim, particles dim, mv channels, multivector dim)

    :param fourmomenta: size: (number of particles, 4)
    :type fourmomenta: torch.Tensor
    :return: Multivector with the correct dimensions
    :rtype: Tensor

    """
    fourmomenta = fourmomenta.unsqueeze(-2)  # (num particles, 1, 4)
    mv = embed_vector(fourmomenta)  # (num particles, 1, 16)
    return mv.unsqueeze(0)  # (1, num particles, 1, 16)

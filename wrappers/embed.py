"""
Functions to embed particle four momenta into the GA multivectors
"""

from typing import Literal

import torch
from lgatr import embed_scalar, embed_vector


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


def to_multivector_parametrized(
    fourmomenta: torch.Tensor,
    theta: torch.Tensor,
    ptr: torch.Tensor,
    mode: Literal["tokens", "channels"],
) -> torch.Tensor:
    """
    Embed the four momenta together with the theory parameters in multivectors

    :param fourmomenta: (num particles, 4)
    :type fourmomenta: torch.Tensor
    :param theta: (batch size, theta dim)
    :type theta: torch.Tensor
    :param ptr: (batch size + 1,)
    :type ptr: torch.Tensor
    :param mode: Mode to encode the theory parameters in the multivectors
    :type mode: Literal["tokens", "channels"]
    :return: Parametrized multivectors
    :rtype: Tensor
    """
    # TODO: Embed paramters either as:
    #   1. Extra multivector channels (one for each dimension of the parameter vector)
    #   this is repeated across the "particles dimension", so each particle gets an
    #   associated set of parameter multivectors (DIRTIER, EASIER)
    #   2. Extra (global) tokens (one for each dimension of the parameter vector)
    #   This are prepended as global tokens for each event (CLEANER, HARDER)
    #   Requires changing the attention mask, event indexing and pooling logic
    mvs = to_multivector(fourmomenta)  # (1, num_particles, 1, 16)

    # Make sure `ptr` is of integer type
    ptr = ptr.to(dtype=torch.long)
    b, n, c, _ = mvs.shape

    if mode == "channels":
        theta_dim = theta.shape[1]

        theta = theta.repeat_interleave(
            ptr[1:] - ptr[:-1], dim=0
        )  # (num_particles, theta_dim)

        assert theta.size() == (n, theta_dim)

        theta_mvs = embed_scalar(theta.unsqueeze(-1))  # (num_particles, theta_dim, 16)
        theta_mvs = theta_mvs.unsqueeze(0)  # (1, num_particles, theta_dim, 16)

        multivectors = torch.cat((mvs, theta_mvs), dim=-2)

        return multivectors  # (batch, particles, theta_dim + 1, 16)

    elif mode == "tokens":
        theta_dim = theta.shape[1]
        token_list = []
        start = 0

        for i in range(len(ptr) - 1):
            end = ptr[i + 1]

            # particles for this event
            part = mvs[:, start:end]

            # theta for this event
            theta_event = theta[i]  # (theta_dim)

            theta_mv = embed_scalar(theta_event.unsqueeze(-1))
            theta_mv = theta_mv.unsqueeze(0)  # (1, theta_dim, 16)

            # reshape to token format
            theta_mv = theta_mv.unsqueeze(2)  # (1, theta_dim, 1, 16)

            # prepend tokens
            tokens = torch.cat((theta_mv, part), dim=1)
            token_list.append(tokens)

            start = end

        multivectors = torch.cat(token_list, dim=1)

        return multivectors
    else:
        raise ValueError(f"Invalid mode {mode}")

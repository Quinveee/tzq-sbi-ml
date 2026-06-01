"""Helper functions"""

from typing import Literal

import torch
from torch.nn.attention import SDPBackend


def ptr2index(ptr: torch.Tensor, mode: Literal["tokens","channels"], theta_dim: int = 0):
    """
    Turns pointer object into repeated indices for each event

    E.g. [0, 2, 4, 5] -> [0, 0, 1, 1, 2]

    :param ptr: Pointer object
    :type ptr: torch.Tensor
    :return: Index object
    :rtype: Tensor
    """
    ptr = ptr.to(dtype=torch.long)
    num_events = len(ptr) - 1

    if mode == "channels":

        return torch.arange(num_events, device=ptr.device).repeat_interleave(
            ptr[1:] - ptr[:-1]
        )

    elif mode == "tokens":
        particles_per_event = ptr[1:] - ptr[:-1]
        tokens_per_event = particles_per_event + theta_dim

        return torch.arange(num_events, device=ptr.device).repeat_interleave(
            tokens_per_event
        )

    else:
        raise ValueError(f"Invalid mode {mode}")


def att_mask(index: torch.Tensor) -> torch.Tensor:
    """
    Return block diagonal matrix (N_particles, N_particles)
    to mask events

    ::note:: In the future we will store this more efficiently

    :param index: Index object constructed from event pointer
    :type index: torch.Tensor
    :return: Block diagonal masking matrix
    :rtype: Tensor
    """
    return (index.unsqueeze(0) == index.unsqueeze(1)).to(torch.bool)


def derive_valid_mask(ptr: torch.Tensor, L_max: int) -> torch.Tensor:
    """
    Build a (B, L_max) bool mask from a CSR-style ptr. True marks real tokens.
    Used as a fallback when callers pass ptr but not an explicit valid_mask.
    """
    ptr = ptr.to(dtype=torch.long)
    lengths = ptr[1:] - ptr[:-1]
    return (
        torch.arange(L_max, device=ptr.device).unsqueeze(0) < lengths.unsqueeze(1)
    )


def padded_to_flat(
    padded: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    Inverse of `flat_to_padded`: extract real tokens from a padded tensor
    using the validity mask. Returns shape (N_total, E). Order follows the
    event-major, position-major layout (i.e. matches a ptr-style flatten).
    """
    return padded[valid_mask]


def flat_to_padded(
    x: torch.Tensor, ptr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a flat token tensor (N_total, E) into a padded batch (B, L_max, E)
    plus a per-token validity mask (B, L_max) where True marks real tokens and
    False marks padding. Padding lets us run batched attention with
    O(B * L_max^2) compute instead of O((B*L_avg)^2) under a block-diagonal
    mask.

    :param x: Flat tokens of shape (N_total, E).
    :param ptr: CSR-style event boundary indices of length B+1.
    :return: (padded tensor, valid_mask)
    """
    ptr = ptr.to(dtype=torch.long, device=x.device)
    lengths = ptr[1:] - ptr[:-1]
    B = int(lengths.numel())
    L_max = int(lengths.max().item()) if B > 0 else 0
    N, E = x.shape

    event_idx = torch.repeat_interleave(
        torch.arange(B, device=x.device), lengths
    )
    pos_in_event = torch.arange(N, device=x.device) - ptr[:-1][event_idx]

    padded = x.new_zeros(B, L_max, E)
    padded[event_idx, pos_in_event] = x

    valid_mask = torch.zeros(B, L_max, device=x.device, dtype=torch.bool)
    valid_mask[event_idx, pos_in_event] = True
    return padded, valid_mask


def get_backends(force_math: bool = False):
    """
    Return available attention backends

    :param force_math: Whether only math backend should be used
    :type force_math: bool
    """
    return [SDPBackend.MATH] + (
        [
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
        if not force_math
        else []
    )

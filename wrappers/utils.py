"""Helper functions"""

import torch
from torch.nn.attention import SDPBackend


def ptr2index(ptr: torch.Tensor) -> torch.Tensor:
    """
    Turns pointer object into repeated indices for each event

    E.g. [0, 2, 4, 5] -> [0, 0, 1, 1, 2]

    :param ptr: Pointer object
    :type ptr: torch.Tensor
    :return: Index object
    :rtype: Tensor
    """
    ptr = ptr.to(dtype=torch.long)
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


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

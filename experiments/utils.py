"""Util functions"""

from dataclasses import fields

import torch

DEVICES = {"gpu": torch.device("cuda"), "cpu": torch.device("cpu")}
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def device(key):
    """Match device string with torch.device"""
    return DEVICES[key]


def dtype(key):
    """Match dtype string with torch.dtype"""
    return DTYPES[key]


def to_device(*args: torch.Tensor, device: torch.device, **kwds) -> list[torch.Tensor]:
    """
    Sends every tensor passed to specified device

    :param args: Tensors
    :type args: torch.Tensor
    :param device: Torch device
    :type device: torch.device
    :param kwds: Additional keyword arguments
    :return: Moved tensors
    :rtype: list[Tensor]
    """
    return [elem.to(device, **kwds) for elem in args]


def to_fields(dcls, **kwargs):
    """
    Sends all attributes in a dataclass that are torch tensors
    to the device specified in kwargs, and casts them to the dtype
    passed in kwargs

    :param dcls: Dataclass object which tensors are to be processed
    :param kwargs: Key word arguments for torch.Tensor.to()
    """
    for f in fields(dcls):
        if f.type is torch.Tensor:
            setattr(dcls, f.name, getattr(dcls, f.name).to(**kwargs))

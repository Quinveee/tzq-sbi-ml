"""
MLP wrappers
"""

from abc import ABC, abstractmethod
import torch
from .base_wrapper import BaseWrapper

class MLPWrapper(BaseWrapper, ABC):
    """
    Base MLP Wrapper
    """

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "MLP"
        super().__init__(*args, **kwds)

    @abstractmethod
    def embed_x(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor, **embedding_kargs) -> torch.Tensor:
        """
        Delegate embedding to subclasses and pass it to the network

        :param x: Event features
        :type x: torch.Tensor
        :param embedding_kargs: Additional embedding keyword arguments
        :return: Forwarded tensor
        :rtype: Tensor
        """
        embedding = self.embed_x(x, **embedding_kargs)
        return self.net(embedding)


class LocalMLPWrapper(MLPWrapper):
    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do nothing, event features are the embedding

        :param x: Event features
        :type x: torch.Tensor
        :return: Embedding vector
        :rtype: Tensor
        """
        return x


class ParametrizedMLPWrapper(MLPWrapper):
    def embed_x(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Just concatenate parameter vector and event features

        :param x: Particle features
        :type x: torch.Tensor
        :param theta: Theory parameter vector
        :type theta: torch.Tensor
        :return: Embedding vector
        :rtype: Tensor
        """
        return torch.cat((theta, x), dim=1)

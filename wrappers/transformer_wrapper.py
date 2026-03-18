"""
Transormer architecture wrappers
"""

from abc import ABC, abstractmethod

import torch
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .utils import att_mask, get_backends, ptr2index


class BaseTransformerWrapper(BaseWrapper, ABC):
    """
    Base Transormer architecture wrapper
    """

    def __init__(self, *args, **kwds):
        kwds["key"] = "Transformer"
        super().__init__(*args, **kwds)

    @abstractmethod
    def embed(self, *args, **kwds):
        pass

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs={},
    ) -> torch.Tensor:
        """
        Forward wrapper for the Transformer

        :param particles: Particles four momenta
        :type particles: torch.Tensor
        :param ptr: Event pointer
        :type ptr: torch.Tensor
        :param force_math: Whether to force non-efficient SA backends
        :type force_math: bool
        :param embedding_kwargs: Additional embedding keyword arguments
        :return: Forwarder tensor
        :rtype: Tensor

        """
        backends = get_backends(force_math)

        # Transfomer already implements a version akin to tokens, so we use 'channels' as default mode 
        index = ptr2index(ptr, mode="channels", theta_dim=embedding_kwargs.get("theta_dim", 0))
        attention_mask = att_mask(index)

        # Delegate embedding to subclasses
        tokens = self.embed(particles, **embedding_kwargs)

        # Just use allowed self-attention backends
        with sdpa_kernel(backends):
            out = self.net(tokens, attn_mask=attention_mask)

        # Here `dim=0` represents the particles dimension
        # We "scatter" the resulting batch using the event pointer
        # and assume a properly set linear output head to match the
        # desired output dimensions
        return scatter(src=out, index=index, dim=0, reduce="mean")


class LocalTransformerWrapper(BaseTransformerWrapper):
    def embed(self, tokens: torch.Tensor, **kwds) -> torch.Tensor:
        """
        Do nothing

        :param tokens: Particle four momenta
        :type tokens: torch.Tensor
        :param kwds:
        :return: Same particle four momenta
        :rtype: torch.Tensor
        """
        return tokens


class ParametrizedTransformerWrapper(BaseTransformerWrapper):
    def embed(
        self, particles: torch.Tensor, theta: torch.Tensor, ptr: torch.Tensor, **kwds
    ) -> torch.Tensor:
        """
        Concatenate particles fourmomenta with theory parameters vector
        Repeat the same theory parameter vector for each particle in the same event

        :param particles: Particles fourmomenta with size: (num particles, 4)
        :type particles: torch.Tensor
        :param theta: Theory parameters vector with size: (batch size, theta dim)
        :type theta: torch.Tensor
        :param ptr: Event pointer with size: (batch size + 1,)
        :type ptr: torch.Tensor
        :param kwds: Description
        :return: Concatenated tensors with size: (num particles, 4 + theta dim)
        :rtype: Tensor

        """
        n, e = particles.shape
        theta_dim = theta.shape[-1]

        ptr = ptr.to(dtype=torch.long)

        theta = theta.repeat_interleave(ptr[1:] - ptr[:-1], dim=0)

        assert theta.size() == (n, theta_dim)

        tokens = torch.cat((theta, particles), dim=-1)

        assert tokens.size() == (n, e + theta_dim)

        return tokens


class LocalTransformerFeaturesWrapper(LocalTransformerWrapper):
    """Transformer wrapper for feature-level local score regression."""

    @staticmethod
    def _to_feature_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split each event feature vector into one scalar token per feature."""
        batch_size, n_features = x.shape
        ptr = torch.arange(
            0,
            (batch_size + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        tokens = x.reshape(-1, 1)
        return tokens, ptr

    def forward(self, x: torch.Tensor, force_math: bool = False, embedding_kwargs={}):
        tokens, ptr = self._to_feature_tokens(x)
        return super().forward(
            particles=tokens,
            ptr=ptr,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )


class ParametrizedTransformerFeaturesWrapper(ParametrizedTransformerWrapper):
    """Transformer wrapper for feature-level ratio regression."""

    @staticmethod
    def _to_feature_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_features = x.shape
        ptr = torch.arange(
            0,
            (batch_size + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        tokens = x.reshape(-1, 1)
        return tokens, ptr

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs={},
    ):
        tokens, ptr = self._to_feature_tokens(x)
        embedding_kwargs = {"theta": theta, "ptr": ptr, **embedding_kwargs}
        return super().forward(
            particles=tokens,
            ptr=ptr,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )

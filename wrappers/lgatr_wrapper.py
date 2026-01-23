"""
LGATr wrappers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Literal, Optional

from hydra.utils import instantiate
from lgatr import extract_scalar
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .decorators import filter_empty_tensor_warning
from .embed import to_multivector, to_multivector_parametrized
from .utils import att_mask, get_backends, ptr2index

if TYPE_CHECKING:
    import torch.nn as nn
    from omegaconf import DictConfig
    from torch import Tensor


class BaseLGATrWrapper(BaseWrapper, ABC):
    """Base LGATr wrapper"""

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "LGATr"
        super().__init__(*args, **kwds)
        self.net = self.init_net(self.net)

    @filter_empty_tensor_warning
    def init_net(self, net: DictConfig) -> nn.Module:
        """
        Inits the LGATr module according to its configuration object

        :param net: Config description of the LGATr module
        :type net: DictConfig
        :return: Torch LGATr module
        :rtype: Module
        """
        return instantiate(net)

    @abstractmethod
    def output(self, multivectors, index, scalars=None):
        pass

    @abstractmethod
    def embed_mv(self, *args, **kwargs):
        pass

    def forward(
        self,
        particles: Tensor,
        ptr: Tensor,
        scalars: Optional[Tensor] = None,
        force_math: bool = False,
        embedding_kwargs: Dict = {},
    ) -> Tensor:
        """
        Forward for any LGATr wrapper. Let subclasses decide how to embed
        the particles' fourmomenta into multivectors as well as how to
        extract the output from the forwarded tensors.

        :param self: Description
        :param particles: Description
        :type particles: Tensor
        :param ptr: Description
        :type ptr: Tensor
        :param scalars: Description
        :type scalars: Optional[Tensor]
        :param force_math: Description
        :type force_math: bool
        :param embedding_kwargs: Description
        :type embedding_kwargs: Dict
        :return: Forwarded multivectors and scalars
        :rtype: Tesor
        """
        # If I want to compute the gradient of the ouitput w.r.t. the
        # parameters I cannot use efficient backends for self attention
        backends = get_backends(force_math)

        # Create masking matrix for self-attention
        index = ptr2index(ptr)
        attention_mask = att_mask(index)

        # Embed fourmomenta
        mv = self.embed_mv(particles, **embedding_kwargs)

        # Run forward using only allowed backends
        with sdpa_kernel(backends):
            # out_mv.size() -> (batch_idx, particles, out_mv_channels, 16)
            out_mv, out_s = self.net(
                multivectors=mv, scalars=scalars, attn_mask=attention_mask
            )

        return self.output(multivectors=out_mv, scalars=out_s, index=index)


class LocalLGATrWrapper(BaseLGATrWrapper):

    def embed_mv(self, particles: Tensor, theta_dim: int) -> Tensor:
        """
        For the local experiment, just repeat the embedded multivector
        `theta_dim` times along the multivector channel dimension

        :param particles: Particles fourmomenta
        :type particles: Tensor
        :param theta_dim: Theory parameter dimension
        :type theta_dim: int
        :return: Multivectors
        :rtype: Tesor

        """
        return to_multivector(particles).repeat(1, 1, theta_dim, 1)

    def output(
        self, multivectors: Tensor, index: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """
        Reduce the multivector batch of each event taking the mean over
        the particles dimension `(dim=1)` and select as ouptut the scalar
        component of the resulting averaged multivectors
        The number of multivector channels is matched with the number of score
        components (see `embed_mv`)

        :param multivectors: Forwarded multivectors
        :type multivectors: Tensor
        :param index: Event index
        :type index: Tensor
        :param scalars: Scalar channels
        :type scalars: Optional[Tensor]
        :return: Forwarded multivectors
        :rtype: Tesor
        """
        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")

        # I assume there are as many output multivector channels as
        # number of dimensions of the score vector to be regressed
        return extract_scalar(out)[0, :, :, 0]  # (batch_size, out_mv_channels)


class ParametrizedLGATrWrapper(BaseLGATrWrapper):
    def embed_mv(
        self,
        particles: Tensor,
        theta: Tensor,
        ptr: Tensor,
        mode: Literal["tokens", "channels"],
    ) -> Tensor:
        """
        Embed the particles' fourmomenta in multivectors and parametrize those
        with the corresponding values for the theory parameters

        :param particles: Particles four momenta
        :type particles: Tensor
        :param theta: Theory parameters
        :type theta: Tensor
        :param ptr: Event pointer
        :type ptr: Tensor
        :param mode: Multivector parametrization mode
        :type mode: Literal["tokens", "channels"]
        :return: Multivectors containing embedded fourmomenta
        :rtype: Tensor
        """
        return to_multivector_parametrized(particles, theta, ptr, mode)

    def output(
        self, multivectors: Tensor, index: Tensor, scalars: Optional[Tensor] = None
    ) -> Tensor:
        """
        Reduce the multivector batch of each event taking the mean over
        the particles dimension `(dim=1)` and select as ouptut the scalar
        component of the multivector corresponding to the first channel.

        :param self: Description
        :param multivectors: Description
        :type multivectors: Tensor
        :param index: Description
        :type index: Tensor
        :param scalars: Description
        :type scalars: Optional[Tensor]
        :return: Description
        :rtype: Tensor
        """
        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")

        # I assume that the first multivector channel corresponds to the
        # fourmomenta embedding, and the others to the (scalar) embeddings of
        # the parameter vector. To extract the regressed log likelihood ratio I
        # take the scalar component of the fourmomenta embedding
        return extract_scalar(out)[0, :, :1, 0]  # (batch_size, 1)

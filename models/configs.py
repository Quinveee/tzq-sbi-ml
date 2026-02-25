"""Configuration schemas for shared modules"""

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class _BaseConfig:
    """Base configuration class"""

    @classmethod
    def cast(cls, conf):
        if isinstance(conf, cls):
            return conf
        if isinstance(conf, Mapping):
            return cls(**conf)
        raise NotImplementedError


@dataclass
class MLPConfig(_BaseConfig):
    """MLP module configuration class"""

    k_factor: int
    activation: str
    dim_in: int = 0  # arbitrary as it's overridden with dataclasses.replace() after construction
    dim_out: int = 0  # arbitrary as it's overridden with dataclasses.replace() after construction
    n_hidden: int = 2
    bias: bool = True
    dropout_p: Optional[float] = None


@dataclass
class SAConfig(_BaseConfig):
    """Self Attention module configuration class"""

    emb_size: int = 0  # arbitrary as it's overridden with dataclasses.replace() after construction
    num_heads: int = 8
    bias: bool = True
    dropout_p: Optional[float] = None
    increase_hidden_channels: int = 8
    # Unused parameters
    # multi_query: bool = False
    # head_scale: bool = False

    @property
    def emb_head(self) -> int:
        """Calculate embedding size of each head.
        Check that full embedding size is divisible by specified attention heads
        """
        assert (
            self.emb_size % self.num_heads == 0
        ), f"Embedding size {self.emb_size} not divisible by n. of heads {self.num_heads}"

        return self.emb_size // self.num_heads

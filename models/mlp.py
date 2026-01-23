"""MLP architecture"""

import torch.nn as nn

from .configs import MLPConfig
from .modules.mlp import MLP as MLPModule


class MLP(nn.Module):
    def __init__(
        self,
        n_observables,
        k_hidden,
        hidden_layers,
        dim_out,
        n_parameters=None,
        activation="tanh",
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        dim_in = n_observables + (n_parameters or 0)
        mlp = MLPConfig(
            dim_in=dim_in,
            dim_out=dim_out,
            k_factor=k_hidden,
            n_hidden=hidden_layers,
            activation=activation,
            bias=bias,
            dropout_p=dropout,
        )
        self.net = MLPModule(mlp)

    def forward(self, inputs):
        return self.net(inputs)

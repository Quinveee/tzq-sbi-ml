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
        self._activation_name = activation
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
        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier for saturating nonlinearities (tanh/sigmoid), Kaiming for
        # (approximately) rectified ones (relu/gelu). PyTorch's default init
        # for nn.Linear is Kaiming-uniform with a=sqrt(5), which is not tuned
        # for any of these activations.
        act = (self._activation_name or "").lower()
        if act in ("tanh", "sigmoid"):
            gain = nn.init.calculate_gain(act)
            init_hidden = lambda w: nn.init.xavier_uniform_(w, gain=gain)
        elif act in ("relu", "gelu"):
            init_hidden = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")
        else:
            init_hidden = nn.init.xavier_uniform_

        linears = [m for m in self.net.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            init_hidden(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        # Output head: no nonlinearity follows -> plain Xavier with gain 1.
        nn.init.xavier_uniform_(linears[-1].weight, gain=1.0)
        if linears[-1].bias is not None:
            nn.init.zeros_(linears[-1].bias)

    def forward(self, inputs):
        return self.net(inputs)

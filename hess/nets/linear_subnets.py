import torch
from torch import nn
from .linear_subnet_layers import SubLayerLinear, MaskedLayerLinear

class SubNetLinear(nn.Module):
    """
    Small MLP
    """
    def __init__(self, in_dim, out_dim, k=16,
                 n_layers=5,
                activation=nn.ReLU(), bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        module = nn.ModuleList()

        module.append(SubLayerLinear(in_dim, k, bias=bias))
        module.append(activation)

        for ll in range(n_layers-1):
            module.append(SubLayerLinear(k, k, bias=bias))
            module.append(activation)

        module.append(SubLayerLinear(k, k, bias=bias))
        module.append(activation)
        module.append(SubLayerLinear(k, out_dim, bias=bias))
        self.sequential = nn.Sequential(*module)

    def forward(self,x):
        return self.sequential(x)


class MaskedNetLinear(nn.Module):
    """
    Small MLP
    """
    def __init__(self, in_dim, out_dim, k=16,
                 n_layers=5,
                activation=nn.ReLU(), bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        module = nn.ModuleList()

        module.append(MaskedLayerLinear(in_dim, k, bias=bias))
        module.append(activation)

        for ll in range(n_layers-1):
            module.append(MaskedLayerLinear(k, k, bias=bias))
            module.append(activation)

        module.append(MaskedLayerLinear(k, k, bias=bias))
        module.append(activation)
        module.append(MaskedLayerLinear(k, out_dim, bias=bias))
        self.sequential = nn.Sequential(*module)

    def forward(self,x):
        return self.sequential(x)

import torch
import math
from torch import nn
from .masked_layer import MaskedLinear

class MaskedNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, x, y, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False, pct_keep=0.6):
        super(MaskedNet, self).__init__()
        self.x = x #inputs in G space
        self.y = y #inputs in D space
        if self.x.ndim == 1:
            self.input_size=1
        else:
            self.input_size = x.size(1) # G = [n x dim x time]
        if self.y.ndim == 1:
            self.output_size = 1 # D = [n x dim x time]
        else:
            self.output_size = self.y.size(1)

        ## initialize the network ##
        module = nn.ModuleList()
        module.append(MaskedLinear(self.input_size, hidden_size, bias=bias,
                                  pct_keep=pct_keep))
        for ll in range(n_hidden-1):
            module.append(activation)
            module.append(MaskedLinear(hidden_size, hidden_size, bias=bias,
                                      pct_keep=pct_keep))


        module.append(activation)
        module.append(MaskedLinear(hidden_size, 1, bias=False,
                                  pct_keep=pct_keep))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

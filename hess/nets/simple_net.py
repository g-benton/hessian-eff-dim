import torch
import math
from torch import nn

class SimpleNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, G_dat, D_dat, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False):
        super(SimpleNet, self).__init__()
        self.G_dat = G_dat #inputs in G space
        self.D_dat = D_dat #inputs in D space
        if self.G_dat.ndim == 1:
            self.input_size=1
        else:
            self.input_size = G_dat.size(1) # G = [n x dim x time]
        if self.D_dat.ndim == 1:
            self.output_size = 1 # D = [n x dim x time]
        else:
            self.output_size = self.D_dat.size(1)

        ## initialize the network ##
        module = nn.ModuleList()
        module.append(nn.Linear(self.input_size, hidden_size, bias=bias))
        for ll in range(n_hidden):
            module.append(activation)
            module.append(nn.Linear(hidden_size, hidden_size, bias=bias))

        module.append(activation)
        module.append(nn.Linear(hidden_size, self.output_size, bias=bias))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

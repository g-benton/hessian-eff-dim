import torch
import math
from torch import nn

class SimpleNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, G_dat, D_dat, hidden_size=10, n_hidden=2):
        super(SimpleNet, self).__init__()
        self.G_dat = G_dat #inputs in G space
        self.D_dat = D_dat #inputs in D space

        self.input_size = G_dat.size(1) # G = [n x dim x time]
        if self.D_dat.ndim == 1:
            self.output_size = 1 # D = [n x dim x time]
        else:
            self.output_size = self.D_dat.size(1)

        ## initialize the network ##
        module = nn.ModuleList()
        module.append(nn.Linear(self.input_size, hidden_size))
        for ll in range(n_hidden):
            module.append(nn.ReLU())
            module.append(nn.Linear(hidden_size, hidden_size))

        module.append(nn.ReLU())
        module.append(nn.Linear(hidden_size, self.output_size))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

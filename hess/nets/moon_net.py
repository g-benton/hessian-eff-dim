import torch
import math
from torch import nn

class MoonNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, x, y, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False):
        super(MoonNet, self).__init__()
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
        module.append(nn.Linear(self.input_size, hidden_size, bias=bias))
        for ll in range(n_hidden-1):
            module.append(activation)
            module.append(nn.Linear(hidden_size, hidden_size, bias=bias))
        
        module.append(nn.Linear(hidden_size, 1, bias=bias))
        module.append(nn.Sigmoid())

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

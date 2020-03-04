import torch
import math
from torch import nn

class SimpleNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, in_dim, out_dim, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False):
        super(SimpleNet, self).__init__()

        ## initialize the network ##
        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, hidden_size, bias=bias))
        for ll in range(n_hidden):
            module.append(activation)
            module.append(nn.Linear(hidden_size, hidden_size, bias=bias))

        module.append(activation)
        module.append(nn.Linear(hidden_size, out_dim, bias=bias))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

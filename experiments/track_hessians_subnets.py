import math
import torch
import numpy as np
import pickle
from torch import nn

import hess
import hess.net_utils as net_utils
from hess.nets import MaskedSubnetLinear
from hess.nets import SubnetLinear

def twospirals(n_points, noise=.5, random_state=920):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


class Subnet(nn.Module):
    """
    Small MLP
    """
    def __init__(self, in_dim, out_dim, k=16,
                 n_layers=5, kernel_size=3,
                activation=nn.ReLU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        module = nn.ModuleList()

        module.append(SubnetLinear(in_dim, k))
        module.append(activation)

        for ll in range(n_layers-1):
            module.append(SubnetLinear(k, k))
            module.append(activation)

        module.append(SubnetLinear(k, k))
        module.append(activation)
        module.append(SubnetLinear(k, out_dim))
        self.sequential = nn.Sequential(*module)

    def forward(self,x):
        return self.sequential(x)


class MaskNet(nn.Module):
    """
    Small MLP
    """
    def __init__(self, in_dim, out_dim, k=16,
                 n_layers=5, kernel_size=3,
                activation=nn.ReLU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        module = nn.ModuleList()

        module.append(MaskedLinear(in_dim, k))
        module.append(activation)

        for ll in range(n_layers-1):
            module.append(MaskedLinear(k, k))
            module.append(activation)

        module.append(MaskedLinear(k, k))
        module.append(activation)
        module.append(MaskedLinear(k, out_dim))
        self.sequential = nn.Sequential(*module)

    def forward(self,x):
        return self.sequential(x)



def main():
    X, Y = twospirals(500, noise=1.3)
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y).unsqueeze(-1)




if __name__ == '__main__':
    main()

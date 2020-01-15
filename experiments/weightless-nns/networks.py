import torch
from torch import nn
import hess
import math
import torch.autograd as autograd
from hess.nets import SubLayerLinear
import torch.nn.functional as F

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class DoubleNet(nn.Module):
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

        module.append(DoubleLayer(in_dim, k, bias=bias))
        module.append(activation)

        for ll in range(n_layers-1):
            module.append(DoubleLayer(k, k, bias=bias))
            module.append(activation)

        module.append(DoubleLayer(k, 2, bias=bias))
        module.append(torch.ReLU())
        module.append(nn.Linear(2,1, bias=False))
        self.sequential = nn.Sequential(*module)
        self.set_weights()

    def forward(self,x):
        return self.sequential(x)


class WeightlessNet(nn.Module):
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

        module.append(SubLayerLinear(k, 2, bias=bias))
        module.append(nn.ReLU)
        module.append(nn.Linear(2,1, bias=False))
        self.sequential = nn.Sequential(*module)
        self.set_weights()

    def forward(self,x):
        return self.sequential(x)

    def set_weights(self):
        for lyr in self.modules():
            if isinstance(lyr, SubLayerLinear):
                new_weights = torch.ones_like(lyr.weight.data)
                lyr.weight.data = new_weights
                if lyr.bias is not None:
                    lyr.bias.data = torch.ones_like(lyr.bias)


        lyr = self.sequential[-1]
        lyr.weight.data = torch.tensor([[-1., 1.]])

class DoubleLayer(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight1 = torch.ones_like(self.weight)
        self.weight2 = torch.ones_like(self.weight) * -1.

        self.scores1 = nn.Parameter(torch.Tensor(self.weight1.size()))
        nn.init.kaiming_uniform_(self.scores1, a=math.sqrt(5))

        self.scores2 = nn.Parameter(torch.Tensor(self.weight2.size()))
        nn.init.kaiming_uniform_(self.scores2, a=math.sqrt(5))
        self.weight = None

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores1(self):
        return self.scores1.abs()

    @property
    def clamped_scores2(self):
        return self.scores2.abs()

    def forward(self, x):
        subnet1 = GetSubnet.apply(self.clamped_scores1, self.prune_rate)
        subnet2 = GetSubnet.apply(self.clamped_scores2, self.prune_rate)
        # print("subnet = ", subnet)

        w1 = self.weight1 * subnet1
        w2 = self.weight2 * subnet2
        x1 = F.linear(x, w1, self.bias)
        x2 = F.linear(x, w2, self.bias)
        # print(x)
        return x1 + x2

def freeze_double_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight1") and m.weight1 is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight1.requires_grad = False
            m.weight2.requires_grad = False
            if m.weight1.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight1.grad = None
            if m.weight2.grad is not None:
                m.weight2.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math
from .conv_type import GetSubnet

class SubLayerLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        # print("subnet = ", subnet)
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        # print(x)
        return x

# class MaskedLayerLinear(nn.Linear):
#     """
#     This is a masked linear layer in which the bias is never masked.
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(MaskedLayerLinear, self).__init__(in_features, out_features, bias=bias)
#         self.mask = torch.ones_like(self.weight)

#     def forward(self, input):
#         new_weight = self.weight * self.mask
#         return F.linear(input, new_weight,
#                         self.bias)

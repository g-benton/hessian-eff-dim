import math
import torch
import torch.nn as nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MaskedLayer(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, pct_keep=0.6):
        super(MaskedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.has_bias = bias
        if self.has_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        dist = torch.distributions.Bernoulli(pct_keep)
        self.mask = dist.sample(sample_shape=torch.Size(self.weight.shape))
        if self.has_bias:
            self.bias_mask = dist.sample(sample_shape=torch.Size(self.bias.shape))

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        if self.has_bias:
            return F.linear(input, self.weight * self.mask,
                            self.bias * self.bias_mask)
        else:
            return F.linear(input, self.weight * self.mask,
                            None)



    def extra_repr(self):
        return 'iln_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

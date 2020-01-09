import math
import torch
import torch.nn as nn
from torch.nn import Module, init, Linear, Conv2d
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class MaskedLinear(Linear):
    #__constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, pct_keep=0.6):
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        self.has_bias = bias

        dist = torch.distributions.Bernoulli(pct_keep)
        self.mask = dist.sample(sample_shape=torch.Size(self.weight.shape))
        # if self.has_bias:
        #     self.bias_mask = dist.sample(sample_shape=torch.Size(self.bias.shape))

    def forward(self, input):
#         if self.weight.device is not self.mask.device:
#             self.mask = self.mask.to(self.weight.device)
        #     if self.has_bias:
        #         self.bias_mask = self.bias_mask.to(self.bias.device)

        if self.has_bias:
            return F.linear(input, self.weight * self.mask,
                            self.bias)
        else:
            return F.linear(input, self.weight * self.mask,
                            None)

class MaskedConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, pct_keep=0.6, bias=True, *args, **kwargs):
        
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, bias=bias, *args, **kwargs)
        
        self.has_bias = bias

        dist = torch.distributions.Bernoulli(pct_keep)
        self.mask = dist.sample(sample_shape=torch.Size(self.weight.shape))
        #if self.has_bias:
            # self.bias_mask = dist.sample(sample_shape=torch.Size(self.bias.shape))


    def conv2d_forward(self, input, weight):
        # if weight.device is not self.mask.device:
        #     self.mask = self.mask.to(weight.device)
        #     if self.has_bias:
        #         self.bias_mask = self.bias_mask.to(self.bias.device)
        #if self.has_bias:
        # else:
        #     if self.padding_mode == 'circular':
        #         expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
        #                             (self.padding[0] + 1) // 2, self.padding[0] // 2)
        #         return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
        #                         weight * self.mask, self.bias, self.stride,
        #                         _pair(0), self.dilation, self.groups)
        #     return F.conv2d(input, weight * self.mask, self.bias, self.stride,
        #                     self.padding, self.dilation, self.groups)      
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight * self.mask, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
      
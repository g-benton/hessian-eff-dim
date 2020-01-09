import torch
import time
import numpy as np
#import hess
from .nets import SubLayerLinear, GetSubnet, MaskedLinear
from torch import nn

def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None

def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")


def get_mask_from_subnet(model):
    mask_list = []
    for lyr in model.modules():
        if isinstance(lyr, SubLayerLinear):
            subnet = GetSubnet.apply(lyr.clamped_scores, lyr.prune_rate)
            mask_list.append(subnet)

    return mask_list

def apply_mask(model, mask):
    mask_ind = 0
    for lyr in model.modules():
        if hasattr(lyr, "mask"):
            if lyr.mask.device is not mask[mask_ind].device:
                lyr.mask = lyr.mask.to(mask[mask_ind].device)

            lyr.mask.data.mul_(0.).add_(mask[mask_ind])
            mask_ind += 1

    print("==> Applied Mask")


def get_weights_from_subnet(model):
    weight_list = []
    for lyr in model.modules():
        if isinstance(lyr, SubLayerLinear):
            weight_list.append(lyr.weight.data)
            if lyr.bias is not None:
                weight_list.append(lyr.bias.data)

    return weight_list

def apply_weights(model, weights):
    wght_ind = 0
    for lyr in model.modules():
        if isinstance(lyr, MaskedLinear):
            lyr.weight.data.mul_(0.).add_(weights[wght_ind].data)
            #lyr.weight.data = weights[wght_ind]
            wght_ind += 1
            if lyr.bias is not None:
                #lyr.bias.data = weights[wght_ind]
                lyr.bias.data.mul_(0.).add_(weights[wght_ind].data)
                wght_ind += 1

    print("==> Applied Weights")

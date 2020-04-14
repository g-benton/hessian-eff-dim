import math
import torch
import torchvision
import hess
from hess.nets import ConvNetDepth
import torchvision
from torchvision import transforms

## model sizes ##
depths = torch.arange(9)
widths = torch.arange(4, 65, 4)

num_classes = 100
use_cuda = torch.cuda.is_available()

for d_ind, dpth in enumerate(depths):
    for w_ind, wdth in enumerate(widths):
        depth = dpth.item()
        width = wdth.item()

        print("depth ", depth, " width ", width, " starting")
        model = ConvNetDepth(num_classes=num_classes, c=width, max_depth=depth)
        if use_cuda:
            model = model.cuda()

        fpath = '/misc/vlgscratch4/WilsonGroup/greg_b/data/'
        fname = "depth_" + str(depth) + "_width_" + str(width) + "_checkpt.pt"

        chckpt = torch.load(fpath + fname)
        model.load_state_dict(chckpt['state_dict'])

import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
import pickle
import argparse
import os, sys
import time

from hess import data
import hess.nets as models
from hess.nets import BasicConv as Net
from parser import parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from compute_loss_surface import get_loss_surface

def gram_schmidt(vector, basis):
    n_base = basis.shape[-1]
    for bb in range(n_base):
        vector = vector - vector.dot(basis[:, bb]).div(basis[:, bb].norm()) * basis[:, bb]
        vector = vector.div(vector.norm())

    return vector

def main():
    args = parser()
    use_cuda = torch.cuda.is_available()


    ## load in everything ##
    fpath = "./"

    model = Net()
    saved_pars = torch.load(fpath + "saved_model.pt")
    if use_cuda:
        model = model.cuda()

    evecs = torch.load(fpath + "top_evecs.pt")
    evals = torch.load(fpath + "top_evals.pt")

    ## get training data ##
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    criterion = nn.CrossEntropyLoss()

    ## compute loss surfaces ##
    fname = "high_loss_" + str(args.range) + "_" + str(args.n_pts)
    fname = fname.replace(".", "_")
    high_loss = get_loss_surface(evecs[:, -10:], model, trainloader,
                            criterion, rng=args.range, n_pts=args.n_pts, use_cuda=use_cuda)

    torch.save(high_loss, fpath + fname)
    print("High Loss Done")
    fname = "low_loss_" + str(args.range) + "_" + str(args.n_pts)
    fname = fname.replace(".", "_")

    v1 = gram_schmidt(torch.randn(evecs.shape[0]).cuda(), evecs).unsqueeze(-1)
    v2 = gram_schmidt(torch.randn(evecs.shape[0]).cuda(), evecs).unsqueeze(-1)
    low_basis = torch.cat((v1, v2), -1)
    low_loss = get_loss_surface(low_basis, model, trainloader,
                            criterion, rng=args.range, n_pts=args.n_pts, use_cuda=use_cuda)
    torch.save(low_loss, fpath + fname)
    print("Low Loss Done")
    fname = "full_loss_" + str(args.range) + "_" + str(args.n_pts)
    fname = fname.replace(".", "_")
    full_loss = get_loss_surface(torch.randn_like(low_basis).cuda(), model, trainloader,
                            criterion, rng=args.range, n_pts=args.n_pts, use_cuda=use_cuda)

    torch.save(full_loss, fpath + fname)
    print("Full Loss Done")

if __name__ == '__main__':
    main()

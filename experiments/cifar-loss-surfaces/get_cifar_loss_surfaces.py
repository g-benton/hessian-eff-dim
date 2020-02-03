import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from compute_loss_surface import get_loss_surface
from min_max_evals import min_max_hessian_eigs
from hess.utils import get_hessian_eigs
import matplotlib.pyplot as plt
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    use_cuda =  torch.cuda.is_available()

    model = Net()
    criterion = torch.nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()

    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)


    ## Super Trainer ##
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    fpath = "./loss-surfaces/"
    fname = "saved_model.pt"
    torch.save(model.state_dict, fpath + fname)

    output = min_max_hessian_eigs(model, trainloader, criterion,
                                  3, 25, use_cuda=use_cuda)

    (pos_evals, pos_evecs, neg_evals, neg_evecs) = output
    fname = "pos_evecs.pt"
    torch.save(pos_evecs, fpath + fname)

    fname = "neg_evecs.pt"
    torch.save(neg_evecs, fpath + fname)

    high_loss = get_loss_surface(pos_evecs, model, trainloader,
                            criterion, rng=1., n_pts=25, use_cuda=use_cuda)
    fname = "high_loss.pt"
    torch.save(high_loss, fpath + fname)

    low_loss = get_loss_surface(neg_evecs, model, trainloader,
                            criterion, rng=1., n_pts=25, use_cuda=use_cuda)
    fname = "low_loss.pt"
    torch.save(low_loss, fpath + fname)


    n_pars = sum(p.numel() for p in model.parameters())
    all_loss = get_loss_surface(torch.eye(n_pars), model, trainloader,
                            criterion, rng=1., n_pts=25, use_cuda=use_cuda)
    fname = "all_loss.pt"
    torch.save(all_loss, fpath + fname)

if __name__ == '__main__':
    main()

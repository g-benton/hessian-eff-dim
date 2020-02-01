

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

if __name__ == '__main__':
    use_cuda =  torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(4)


    # In[4]:


    model = Net()
    model.load_state_dict(torch.load("./model.pt"))

    # In[5]:


    if use_cuda:
        model = model.cuda()

    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    dataiter = iter(testloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



    evals, evecs = get_hessian_eigs(loss=criterion,
                         model=model, use_cuda=True, n_eigs=200,
                         loader=trainloader, evals=True)




    fpath = "./"

    fname = "cifar_evals_200.pt"
    torch.save(evals, fpath + fname)

    fname = "cifar_evecs_200.pt"
    torch.save(evecs, fpath + fname)

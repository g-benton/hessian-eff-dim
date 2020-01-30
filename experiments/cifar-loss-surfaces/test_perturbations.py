import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import hess
import hess.utils as utils

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


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

def compute_loss_differences(model, loader, criterion,
                            model_preds, use_cuda=True):
    train_loss = 0.
    train_diffs = 0
    for dd, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print statistics
        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_diffs += torch.sum(predicted.cpu() != model_preds[dd])

    return train_loss, train_diffs

def gram_schmidt(vector, basis):
    n_base = basis.shape[-1]
    for bb in range(n_base):
        vector = vector - vector.dot(basis[:, bb]).div(basis[:, bb].norm()) * basis[:, bb]
        vector = vector.div(vector.norm())

    return vector

def main():

    ## generate model and load in trained instance ##
    use_cuda = torch.cuda.is_available()
    model = Net()

    saved_model = torch.load("./model.pt", map_location=('cpu'))
    model.load_state_dict(saved_model)
    if use_cuda:
        model = model.cuda()


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/datasets/cifar10/', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    ## load in eigenpairs and clean up ##
    fpath = "./"
    fname = "cifar_evecs_200.pt"
    evecs = torch.load(fpath + fname, map_location=("cpu")).squeeze()

    fname = "cifar_evals_200.pt"
    evals = torch.load(fpath + fname, map_location=("cpu"))

    keep = np.where(evals != 1)[0]
    n_evals = keep.size
    evals = evals[keep].numpy()
    evecs = evecs[:, keep].numpy()

    idx = np.abs(evals).argsort()[::-1]
    evals = torch.FloatTensor(evals[idx])
    evecs = torch.FloatTensor(evecs[:, idx])

    pars = utils.flatten(model.parameters())
    n_par = pars.numel()
    par_len = pars.norm()

    criterion = nn.CrossEntropyLoss()

    ## going to need the original model predictions ##
    model_preds = []
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        model_preds.append(predicted.cpu())


    n_scale = 20
    n_trial = 10
    keep_evecs = 20
    scales = torch.linspace(0, 1., n_scale)

    ## Test high curvature directions ##
    high_curve_losses = torch.zeros(n_scale, n_trial)
    n_diff_high = torch.zeros(n_scale, n_trial)
    for ii in range(n_scale):
        for tt in range(n_trial):
            alpha = torch.randn(keep_evecs)
            pert = evecs[:, :keep_evecs].matmul(alpha.unsqueeze(-1)).t()
            pert = scales[ii] * pert.div(pert.norm())
            if use_cuda:
                pert = pert.cuda()
            pert = utils.unflatten_like(pert, model.parameters())

            ## perturb ##
            for i, par in enumerate(model.parameters()):
                par.data = par.data + pert[i]

            ## compute the loss and label diffs ##
            train_loss, train_diff = compute_loss_differences(model, testloader,
                                                              criterion, model_preds)
            high_curve_losses[ii, tt] = train_loss
            n_diff_high[ii, tt] = train_diff

            ## need to reload pars after each perturbation ##
            model.load_state_dict(saved_model)

        ## just to track progress ##
        print("high curve scale {} of {} done".format(ii, n_scale))

    ## save the high curvature results ##
    fpath = "./"
    fname = "high_curve_losses_test.pt"
    torch.save(high_curve_losses, fpath + fname)

    fname = "n_diff_high_test.pt"
    torch.save(n_diff_high, fpath + fname)
    print("all high curvature done \n\n")

    ## go through the low curvature directions ##
    low_curve_losses = torch.zeros(n_scale, n_trial)
    n_diff_low = torch.zeros(n_scale, n_trial)
    for ii in range(n_scale):
        for tt in range(n_trial):
            alpha = torch.randn(n_par) # random direction
            pert = gram_schmidt(alpha, evecs).unsqueeze(-1).t() # orthogonal to evecs
            pert = scales[ii] * pert.div(pert.norm()) # scaled correctly

            if use_cuda:
                pert = pert.cuda()
            pert = utils.unflatten_like(pert, model.parameters())

            ## go through trainloader and keep track of losses/differences in preds ##
            for i, par in enumerate(model.parameters()):
                par.data = par.data + pert[i]

            ## compute the loss and label diffs ##
            train_loss, train_diff = compute_loss_differences(model, testloader,
                                                              criterion, model_preds)
            low_curve_losses[ii, tt] = train_loss
            n_diff_low[ii, tt] = train_diff

            model.load_state_dict(saved_model)

        print("low curve scale {} of {} done".format(ii, n_scale))

    ## save the high curvature results ##
    fpath = "./"
    fname = "low_curve_losses_test.pt"
    torch.save(low_curve_losses, fpath + fname)

    fname = "n_diff_low_test.pt"
    torch.save(n_diff_low, fpath + fname)

    print("\n\n Gucci.")

if __name__ == '__main__':
    main()

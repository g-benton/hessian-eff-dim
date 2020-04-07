import torch
import math
import copy
import warnings
import torch.nn as nn
import numpy as np
from hess import utils


# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.data = param.data.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p )).item()


def perturb_model(model, sigma, n_pars):
    perturb = torch.randn(n_pars) * sigma
    perturb = utils.unflatten_like(perturb.unsqueeze(0), model.parameters())

    for i, par in enumerate(model.parameters()):
        par.data = par.data + perturb[i]

    return

def compute_accuracy(model, dataloader, n_batch_samples):
    accuracy = 0
    for batch in range(n_batch_samples):
        images, labels = next(iter(dataloader))
        preds = model(images).max(-1)[1]
        accuracy += torch.where(preds == labels)[0].numel()

    return accuracy/(n_batch_samples * dataloader.batch_size)



def sharpness_sigma(model, trainloader, target_deviate=0.1, resample_sigma=10,
                    n_batch_samples=10, n_midpt_rds=20, upper=5., lower=0.,
                    bound_eps=1e-3, discrep_eps=1e-2,
                    use_cuda=False):

    ## compute training accuracy ##
    train_accuracy = compute_accuracy(model, trainloader, n_batch_samples)

    ## store saved pars ##
    saved_pars = model.state_dict()
    n_pars = sum([p.numel() for p in model.parameters()])

    for midpt_iter in range(n_midpt_rds):
        ## for each iteration of midpoint method
        model.load_state_dict(saved_pars)
        midpt = (upper + lower)/2.

        ## compute estimate of error with perturbed parameters
        rnd_errors = torch.zeros(resample_sigma)
        perturb_model(model, midpt, n_pars)
        for rnd in range(resample_sigma):
            rnd_errors[rnd] = compute_accuracy(model, trainloader, n_batch_samples)

        ## how much has perturbation changed
        rnd_error = rnd_errors.mean()
        discrepancy = torch.abs(train_accuracy - rnd_error)

        if ((upper - lower) < bound_eps) or (discrepancy < discrep_eps):
            return midpt

        elif rnd_error > target_deviate:
            ## can cutoff the upper half
            upper = midpt
            # print("cutoff upper\n")
        else:
            ## can cutoff the lower half
            lower = midpt
            # print("cutoff lower\n")


    return midpt

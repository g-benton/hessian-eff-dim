"""
    compute hessian vector products as well as eigenvalues of the hessian
    # copied from https://github.com/tomgoldstein/loss-landscape/blob/master/hess_vec_prod.py
    # code re-written to use gpu by default and then to use gpytorch
"""

import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from hess.utils import flatten, unflatten_like, eval_hess_vec_prod, gradtensor_to_tensor

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(
    net, dataloader, criterion, rank=0, use_cuda=True, verbose=False, nsteps=100,
    return_evecs=False
):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.
        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information
        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)
    nb = len(dataloader)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader=dataloader, use_cuda=True)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        out = gradtensor_to_tensor(net)
        if not use_cuda:
            out = out.cpu()

        return out.unsqueeze(1) / nb

    hess_vec_prod.count = 0
    if verbose and rank == 0:
        print("Rank %d: computing max eigenvalue" % rank)
    if use_cuda:
        device = params[0].device
    else:
        print(params[0].dtype)
        device = torch.zeros(1).device

    # use lanczos to get the t and q matrices out
    pos_q_mat, pos_t_mat = lanczos_tridiag(
        hess_vec_prod,
        nsteps,
        device=device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    pos_eigvals, pos_eigvecs = lanczos_tridiag_to_diag(pos_t_mat)
    print(pos_eigvals)
    # eigenvalues may not be sorted
    maxeig = torch.max(pos_eigvals)

    pos_bases = pos_q_mat @ pos_eigvecs
    if verbose and rank == 0:
        print("max eigenvalue = %f" % maxeig)

    if not return_evecs:
        return maxeig, None, hess_vec_prod.count, pos_eigvals, None, pos_t_mat
    else:
        return maxeig, None, hess_vec_prod.count, pos_eigvals, pos_bases, pos_t_mat

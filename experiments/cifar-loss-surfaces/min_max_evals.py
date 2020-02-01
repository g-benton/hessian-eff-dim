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

from hess.utils import flatten, unflatten_like, eval_hess_vec_prod

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(net, dataloader, criterion,
                         n_top_eigs=3, n_bottom_eigs=50, use_cuda=False):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.
        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            use_cuda: use GPU
    """

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader=dataloader,
                          use_cuda=use_cuda)
        prod_time = time.time() - start_time
        out = gradtensor_to_tensor(net)
        return out.unsqueeze(1)

    hess_vec_prod.count = 0

    # use lanczos to get the t and q matrices out
    pos_q_mat, pos_t_mat = lanczos_tridiag(
        hess_vec_prod,
        n_top_eigs,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    pos_eigvals, pos_eigvecs = lanczos_tridiag_to_diag(pos_t_mat)

    pos_eigvecs = pos_q_mat @ pos_eigvecs

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    # shift = maxeig*.51
    shift = 0.51 * maxeig.item()
    print("Pos Eigs Computed....\n")

    def shifted_hess_vec_prod(vec):
        hvp = hess_vec_prod(vec)
        return -hvp + shift * vec


    # now run lanczos on the shifted eigenvalues
    neg_q_mat, neg_t_mat = lanczos_tridiag(
        shifted_hess_vec_prod,
        n_bottom_eigs,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    neg_eigvals, neg_eigvecs = lanczos_tridiag_to_diag(neg_t_mat)
    neg_eigvecs = neg_q_mat @ neg_eigvecs
    print("Neg Eigs Computed...\n")

    neg_evals = -neg_evals + shift


    #return maxeig, mineig, hess_vec_prod.count, pos_eigvals, neg_eigvals, pos_bases
    return pos_eigvals, pos_eigvecs, neg_eigvals, neg_eigvecs

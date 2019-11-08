import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from hess.utils import flatten, unflatten_like

def max_eigenpairs(net, dataloader, criterion, n_eigs, rank=0
                 use_cuda=torch.cuda.is_available()):]

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        out = gradtensor_to_tensor(net)
        return out.unsqueeze(1)

    pos_q_mat, pos_t_mat = lanczos_tridiag(
        hess_vec_prod,
        100,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    e_vals, e_vecs = lanczos_tridiag_to_diag(pos_t_mat)

    ## GOING TO NEED TO CHANGE E_VECS HERE ##

    return e_vals, e_vecs

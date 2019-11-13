import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from hess.utils import flatten, unflatten_like, gradtensor_to_tensor
from hess.utils import eval_hess_vec_prod

def hessian_eigenpairs(net, criterion,
                 inputs=None, targets=None, dataloader=None,
                 n_eigs=20,
                 use_cuda=torch.cuda.is_available(),
                 verbose=False):

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, inputs=inputs,
                           targets=targets,
                           dataloader=dataloader, use_cuda=use_cuda)
        prod_time = time.time() - start_time
        if verbose:
            print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        out = gradtensor_to_tensor(net)
        return out.unsqueeze(1)

    pos_q_mat, pos_t_mat = lanczos_tridiag(
        hess_vec_prod,
        n_eigs,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    e_vals, e_vecs = lanczos_tridiag_to_diag(pos_t_mat)

    ## GOING TO NEED TO CHANGE E_VECS HERE ##
    e_vecs = pos_q_mat.matmul(e_vecs)

    return e_vals, e_vecs

import torch
import time
import numpy as np
import hess
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from hess.utils import unflatten_like, gradtensor_to_tensor, eval_hess_vec_prod

def get_hessian_evals(loss, model,
                     use_cuda=False, n_eigs=100, train_x=None, train_y=None,
                     loader=None):
    if train_x is not None:
        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

    total_pars = sum(m.numel() for m in model.parameters())

    def hvp(rhs):
        padded_rhs = torch.zeros(total_pars, rhs.shape[-1],
                                 device=rhs.device, dtype=rhs.dtype)

        padded_rhs = unflatten_like(padded_rhs.t(), model.parameters())
        eval_hess_vec_prod(padded_rhs, net=model,
                           criterion=loss, inputs=train_x,
                           targets=train_y, dataloader=loader, use_cuda=use_cuda)
        full_hvp = gradtensor_to_tensor(model, include_bn=True)
        return full_hvp.unsqueeze(-1)

#         print('numpars is: ', numpars)
    if train_x is None:
        data = next(iter(loader))[0]
        if use_cuda:
            data = data.cuda()
        dtype = data.dtype
        device = data.device
    else:
        dtype, device = train_x.dtype, train_x.device

    qmat, tmat = lanczos_tridiag(hvp, n_eigs, dtype=dtype,
                              device=device, matrix_shape=(total_pars,
                              total_pars))
    eigs, t_evals = lanczos_tridiag_to_diag(tmat)

    return eigs, qmat @ t_evals

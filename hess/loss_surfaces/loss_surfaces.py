import math
import torch
import numpy as np
from .. import utils

def get_loss_surface(basis, model,
                    train_x, train_y,
                    loss,
                    rng=0.1, n_pts=25,
                    use_cuda=False, **kwargs):
    """
    note that loss should be a lambda function that just takes in the model!
    """

    start_pars = model.state_dict()
    ## get out the plane ##
    dir1, dir2 = get_plane(basis, **kwargs)

    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    vec_len = torch.linspace(-rng/2., rng/2., n_pts)

    ## loop and get loss at each point ##
    for ii in range(n_pts):
        for jj in range(n_pts):
            perturb = dir1.mul(vec_len[ii]) + dir2.mul(vec_len[jj])
            # print(perturb.shape)
            perturb = utils.unflatten_like(perturb.t(), model.parameters())
            for i, par in enumerate(model.parameters()):
                if use_cuda:
                    par.data = par.data + perturb[i].cuda()
                else:
                    par.data = par.data + perturb[i]

            loss_surf[ii, jj] = loss(model(train_x), train_y)

            model.load_state_dict(start_pars)

    return loss_surf

def get_plane(basis, scale=0.1):
    """
    returns two vectors that define the span of a random plane
    that is in the span of the basis
    """
    n_basis = basis.size(-1)
    wghts = torch.randn(n_basis, 1)
    dir1 = basis.matmul(wghts)

    wghts = torch.randn(n_basis, 1)
    dir2 = basis.matmul(wghts)

    ## now gram schmidt these guys ##
    vu = dir2.squeeze().dot(dir1.squeeze())
    uu = dir1.squeeze().dot(dir1.squeeze())

    dir2 = dir2 - dir1.mul(vu).div(uu)

    ## normalize ##
    dir1 = dir1.div(dir1.norm()) * scale
    dir2 = dir2.div(dir2.norm()) * scale

    return dir1, dir2

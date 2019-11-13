import math
import torch
import numpy as np
from .. import utils
from .loss_surfaces import get_plane

def dataloader_loss_surface(basis, model,
                    dataloader,
                    loss=torch.nn.MSELoss(),
                    rng=0.1, n_pts=25, **kwargs):

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
            perturb = utils.unflatten_like(perturb.t(), model.parameters())
            for i, par in enumerate(model.parameters()):
                par.data = par.data + perturb[i]
            curr_loss = 0
            for inputs, targets in dataloader:
                output = model(inputs)
                curr_loss += loss(output, targets)

            loss_surf[ii, jj] = curr_loss
            model.load_state_dict(start_pars)
            print("point done")

    return loss_surf

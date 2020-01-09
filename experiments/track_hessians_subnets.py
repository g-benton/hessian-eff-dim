import math
import torch
import numpy as np
import pickle
from torch import nn

import hess
import hess.net_utils as net_utils
import hess.utils as utils
from hess.nets import MaskedNetLinear, SubNetLinear
# from hess.nets import MaskedLayerLinear, SubLayerLinear

def twospirals(n_points, noise=.5, random_state=920):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


def main():
    X, Y = twospirals(500, noise=1.3)
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y).unsqueeze(-1)

    ###################################
    ## Set up nets and match weights ##
    ###################################

    n_hidden = 5
    width = 1024

    subnet_model = SubNetLinear(in_dim=2, out_dim=1, n_layers=n_hidden, k=width, bias=False)
    masked_model = MaskedNetLinear(in_dim=2, out_dim=1, n_layers=n_hidden, k=width, bias=False)

    hess.net_utils.set_model_prune_rate(subnet_model, 0.5)
    hess.net_utils.freeze_model_weights(subnet_model)

    weights = net_utils.get_weights_from_subnet(subnet_model)

    net_utils.apply_weights(masked_model, weights)
    mask = net_utils.get_mask_from_subnet(subnet_model)
    net_utils.apply_mask(masked_model, mask)
    mask = utils.flatten(mask)
    print(mask)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        subnet_model = subnet_model.cuda()
        masked_model = masked_model.cuda()

    ######################
    ## Train the Subnet ##
    ######################

    optimizer = torch.optim.Adam(subnet_model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()
    eigs_every = 10
    n_eigs = 100
    eigs_out = []

    for step in range(1000):
        optimizer.zero_grad()
        outputs = subnet_model(train_x)

        loss=loss_func(outputs,train_y)
        print(loss)
        loss.backward()
        optimizer.step()

        if step % eigs_every == 0:
            mask = net_utils.get_mask_from_subnet(subnet_model)
            net_utils.apply_mask(masked_model, mask)
            mask = utils.flatten(mask)

            eigs = utils.get_hessian_eigs(loss_func, masked_model, mask=mask,
                                          n_eigs=n_eigs, train_x=train_x,
                                          train_y=train_y)

            eigs_out.append(eigs)


    fpath = "./saved-subnet-hessian/"
    fname = "subnet_eigs.pkl"

    with open(fpath + fname, 'wb') as f:
        pickle.dump(eigs_out, f)

    fname = "subnet_model.pt"
    torch.save(subnet_model.state_dict(), fpath + fname)

    fname = "masked_model.pt"
    torch.save(masked_model.state_dict(), fpath + fname)


if __name__ == '__main__':
    main()

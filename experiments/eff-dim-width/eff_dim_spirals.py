import math
import torch
import hess
import matplotlib.pyplot as plt
from hess.nets import Transformer
import hess.loss_surfaces as loss_surfaces
import numpy as np
import sklearn.datasets as datasets
import hess.utils as utils
import pickle

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

    ##########################
    ## SET UP TRAINING DATA ##
    ##########################
    X, Y = twospirals(500, noise=1.5)
    train_x, train_y = torch.FloatTensor(X), torch.FloatTensor(Y).unsqueeze(-1)

    test_X, test_Y = twospirals(100, 1.5)
    test_x, test_y = torch.FloatTensor(test_X), torch.FloatTensor(test_Y).unsqueeze(-1)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(2)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        train_x, train_y = train_x.cuda(), train_y.cuda()


    #############################
    ## SOME HYPERS AND STORAGE ##
    #############################
    widths = [i for i in range(5, 6)]
    loss_func = torch.nn.BCEWithLogitsLoss()

    in_dim = 2
    out_dim = 1

    hessians = []
    n_pars = []
    test_errors = []
    ###############
    ## MAIN LOOP ##
    ###############
    for width_ind, width in enumerate(widths):
        model = hess.nets.SimpleNet(in_dim, out_dim, n_hidden=5, hidden_size=20,
                             activation=torch.nn.ELU(), bias=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        n_par = sum(p.numel() for p in model.parameters())
        n_pars.append(n_par)


        ## TRAIN MODEL ##
        for step in range(2000):
            optimizer.zero_grad()
            outputs = model(train_x)

            loss=loss_func(outputs, train_y)
            loss.backward()
            optimizer.step()
        print("model %i trained" % width)

        hessian = torch.zeros(n_par, n_par)
        for pp in range(n_par):
            base_vec = torch.zeros(n_par).unsqueeze(0)
            base_vec[0, pp] = 1.

            base_vec = utils.unflatten_like(base_vec, model.parameters())
            utils.eval_hess_vec_prod(base_vec, model,
                                    criterion=torch.nn.BCEWithLogitsLoss(),
                                    inputs=train_x, targets=train_y)
            if pp == 0:
                output = utils.gradtensor_to_tensor(model, include_bn=True)
                hessian = torch.zeros(output.nelement(), output.nelement())
                hessian[:, pp] = output

            hessian[:, pp] = utils.gradtensor_to_tensor(model, include_bn=True).cpu()

        ## SAVE THOSE OUTPUTS ##
        hessians.append(hessian)
        test_errors.append(loss_func(model(test_x), test_y).item())

    ## SAVE EVERYTHING ##
    fpath = "./"
    fname = "hessians.P"
    with open(fpath + fname, 'wb') as fp:
        pickle.dump(hessians, fp)

    fname = "test_errors.P"
    with open(fpath + fname, 'wb') as fp:
        pickle.dump(test_errors, fp)

    fname = "n_pars.P"
    with open(fpath + fname, 'wb') as fp:
        pickle.dump(n_pars, fp)

    fname = "widths.pt"
    torch.save(widths, fpath + fname)


if __name__ == '__main__':
    main()

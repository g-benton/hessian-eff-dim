import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
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
    X, Y = twospirals(500, noise=1.3)
    train_x = torch.FloatTensor(X)
    train_y = torch.FloatTensor(Y).unsqueeze(-1)

    X, Y = twospirals(100, noise=1.3)
    test_x = torch.FloatTensor(X)
    test_y = torch.FloatTensor(Y).unsqueeze(-1)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(2)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()

    loss_func = torch.nn.BCEWithLogitsLoss()
    lr = 0.01

    n_trials = 200
    n_iters = 1000
    losses = torch.zeros(n_trials, n_iters)
    test_losses = torch.zeros(n_trials, n_iters)
    init_eigs = []
    final_eigs = []
    pct_keep = 0.4
    optim = torch.optim.Adam

    for trial in range(n_trials):
        model = hess.nets.MaskedNet(train_x, train_y, bias=True,
                                n_hidden=5, hidden_size=10,
                                activation=torch.nn.ELU(),
                                pct_keep=pct_keep)

        if use_cuda:
            model = model.cuda()
        mask, perm = hess.utils.mask_model(model, pct_keep, use_cuda)
        keepers = np.array(np.where(mask.cpu() == 1))[0]

        ## compute hessian pre-training ##
        hessian = utils.get_hessian(train_x, train_y, loss=loss_func,
                             model=model, use_cuda=use_cuda)
        sub_hess = hessian[np.ix_(keepers, keepers)]
        e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        init_eigs.append(e_val.real)

        ## train ##
        optimizer=optim(model.parameters(), lr=lr)

        for step in range(n_iters):
            optimizer.zero_grad()
            outputs = model(train_x)

            loss = loss_func(outputs,train_y)
            losses[trial, step] = loss

            test_out = model(test_x)
            test_losses[trial, step] = loss_func(test_out, test_y)

            loss.backward()
            optimizer.step()

        ## compute final hessian ##
        hessian = utils.get_hessian(train_x, train_y, loss=loss_func,
                             model=model, use_cuda=use_cuda)
        sub_hess = hessian[np.ix_(keepers, keepers)]
        e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        final_eigs.append(e_val.real)


        print("model ", trial, " done")

    fpath = "../saved-experiments/"

    fname = "train_losses.pt"
    torch.save(losses, fpath + fname)

    fname = "test_losses.pt"
    torch.save(test_losses, fpath + fname)

    fname = "init_eigs.P"
    with open(fpath + fname, 'wb') as fp:
        pickle.dump(init_eigs, fp)

    fname = "final_eigs.P"
    with open(fpath + fname, 'wb') as fp:
        pickle.dump(final_eigs, fp)

if __name__ == '__main__':
    main()

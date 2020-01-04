import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
import pickle
import argparse
import os, sys

from hess import data
import hess.nets as models
from parser import parser



def main():
    args = parser()
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print("Preparing directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=args.split_classes,
    )



    loss_func = torch.nn.BCEWithLogitsLoss()
    lr = 0.01

    n_trials = 2
    n_iters = 1000
    losses = torch.zeros(n_trials, n_iters)
    init_eigs = []
    final_eigs = []
    pct_keep = 0.4
    optim = torch.optim.SGD

    for trial in range(n_trials):
        print("Preparing model")
        print(*model_cfg.args)
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        model.to(args.device)

        # model = hess.nets.cifar_net(train_x, train_y, bias=True, 
        #                         n_hidden=5, hidden_size=10,
        #                         activation=torch.nn.ELU(),
        #                         pct_keep=pct_keep)

        if use_cuda:
            model = model.cuda()
        mask, perm = hess.utils.mask_model(model, pct_keep, use_cuda)
        keepers = np.array(np.where(mask.cpu() == 1))[0]

        ## compute hessian pre-training ##
        initial_evals = utils.get_hessian_eigs(train_x, train_y,
            loss=loss_func, model=model, mask=mask, use_cuda=use_cuda, n_eigs=100)
        init_eigs.append(initial_evals)
        # hessian = utils.get_hessian(train_x, train_y, loss=loss_func,
        #                      model=model, use_cuda=use_cuda)
        # sub_hess = hessian[np.ix_(keepers, keepers)]
        # e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        # init_eigs.append(e_val.real)

        ## train ##
        optimizer=optim(model.parameters(), lr=lr)

        for step in range(n_iters):
            optimizer.zero_grad()
            outputs = model(train_x)

            loss=loss_func(outputs,train_y)
            losses[trial, step] = loss
            loss.backward()
            optimizer.step()

        ## compute final hessian ##
        hessian = utils.get_hessian(train_x, train_y, loss=loss_func,
                             model=model, use_cuda=use_cuda)
        sub_hess = hessian[np.ix_(keepers, keepers)]
        e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        final_eigs.append(e_val.real)

        print("model ", trial, " done")

    # fpath = "../saved-experiments/"

    # fname = "losses.pt"
    # torch.save(losses, fpath + fname)

    # fname = "init_eigs.P"
    # with open(fpath + fname, 'wb') as fp:
    #     pickle.dump(init_eigs, fp)

    # fname = "final_eigs.P"
    # with open(fpath + fname, 'wb') as fp:
    #     pickle.dump(final_eigs, fp)

if __name__ == '__main__':
    main()

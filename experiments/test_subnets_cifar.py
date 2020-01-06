import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
import pickle
import argparse
import os, sys

import swag.utils as training_utils
from hess import data
import hess.nets as models
from parser import parser

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

def main():
    args = parser()
    args.device = None

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    #loss_func = torch.nn.BCEWithLogitsLoss()
    #lr = 0.01

    n_trials = 1
    #n_iters = 1000
    #losses = torch.zeros(n_trials, n_iters)
    init_eigs = []
    final_eigs = []
    #pct_keep = 0.4
    #optim = torch.optim.SGD
    
    print("Preparing base directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)

    for trial in range(n_trials):
        print("Preparing directory %s" % args.dir+'/trial_'+str(trial))
        
        os.makedirs(args.dir+'/trial_'+str(trial), exist_ok=True)
        with open(os.path.join(args.dir+'/trial_'+str(trial), "command.sh"), "w") as f:
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

        print("Preparing model")
        print(*model_cfg.args)
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        model.to(args.device)

        mask = hess.utils.get_mask(model)
        #mask, perm = hess.utils.mask_model(model, pct_keep, use_cuda)
        keepers = np.array(np.where(mask.cpu() == 1))[0]

        ## compute hessian pre-training ##
        initial_evals = utils.get_hessian_eigs(loss=loss_func, model=model, mask=mask, 
                                               use_cuda=args.cuda, n_eigs=100, loader=loaders['train'])
        init_eigs.append(initial_evals)

        ## train ##
        optimizer=optim(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        criterion = torch.nn.functional.cross_entropy

        for epoch in range(0, args.epochs):
            train_epoch(model, loaders, criterion, optimizer, epoch, 
                    args.epochs, args.eval_freq, args.save_freq, args.dir+'/trial_'+str(trial))        

        ## compute final hessian ##
        hessian = utils.get_hessian_eigs(loss=loss_func,
                             model=model, use_cuda=args.cuda, n_eigs=100, mask=mask,
                             loader=loaders['train'])
        # sub_hess = hessian[np.ix_(keepers, keepers)]
        # e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        # final_eigs.append(e_val.real)

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

def train_epoch(model, loaders, criterion, optimizer, schedule, epoch, end_epoch,
            eval_freq = 1, save_freq = 10, output_dir='./'):

    time_ep = time.time()

    lr = training_utils.schedule(epoch)
    training_utils.adjust_learning_rate(optimizer, lr)
    train_res = training_utils.train_epoch(loaders["train"], model, criterion, optimizer)
    if (
        epoch == 0
        or epoch % eval_freq == eval_freq - 1
        or epoch == end_epoch - 1
    ):
        test_res = training_utils.eval(loaders["test"], model, criterion)
    else:
        test_res = {"loss": None, "accuracy": None}

    if (epoch + 1) % save_freq == 0:
        training_utils.save_checkpoint(
            output_dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )

    time_ep = time.time() - time_ep
    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if __name__ == '__main__':
    main()

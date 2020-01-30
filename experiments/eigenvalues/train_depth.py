import math
import torch
import hess
import hess.utils as utils
import hess.nets
import numpy as np
import pickle
import argparse
import os, sys
import time
import tabulate

import swag.utils as training_utils
import swag
from hess import data
import hess.nets as models
from parser import parser

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

def main():
    args = parser()
    args.device = None

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False

    #loss_func = torch.nn.BCEWithLogitsLoss()
    #lr = 0.01

    n_trials = 1
    #n_iters = 1000
    #losses = torch.zeros(n_trials, n_iters)
    #init_eigs = []
    #final_eigs = []
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
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                               max_depth=args.num_channels)
        model.to(args.device)
        # bad set to for now
        # for m in model.modules():
        #     if isinstance(m, hess.nets.MaskedConv2d) or isinstance(m, hess.nets.MaskedLinear):
        #         if m.mask is not None and m.weight is not None:
        #             m.mask = m.mask.to(m.weight.device)
        #         if m.has_bias:
        #             if m.bias_mask is not None and m.bias is not None:
        #                 m.bias_mask = m.bias_mask.to(m.bias.device)

        # mask = hess.utils.get_mask(model)
        # mask, perm = hess.utils.mask_model(model, pct_keep, use_cuda)
        # keepers = np.array(np.where(mask.cpu() == 1))[0]

        # criterion = torch.nn.functional.cross_entropy

        ## compute hessian pre-training ##
        # initial_evals = utils.get_hessian_eigs(loss=criterion, model=model, mask=mask, 
        #                                        use_cuda=args.cuda, n_eigs=100, loader=loaders['train'])
        # init_eigs.append(initial_evals)

        ## train ##
        optimizer=torch.optim.SGD(model.parameters(), 
                                  lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)

        for epoch in range(0, args.epochs):
            train_epoch(model, loaders, swag.losses.cross_entropy, optimizer, 
                    epoch=epoch, 
                    end_epoch=args.epochs, eval_freq=args.eval_freq, save_freq=args.save_freq, 
                    output_dir=args.dir+'/trial_'+str(trial),
                    lr_init=args.lr_init)        

        ## compute final hessian ##
        # final_evals = utils.get_hessian_eigs(loss=criterion,
        #                      model=model, use_cuda=args.cuda, n_eigs=100, mask=mask,
        #                      loader=loaders['train'])
        # sub_hess = hessian[np.ix_(keepers, keepers)]
        # e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        # final_eigs.append(e_val.real)
        # final_eigs.append(final_evals)

        print("model ", trial, " done")

        # fpath = "../saved-experiments/"

        # fname = "losses.pt"
        # torch.save(losses, fpath + fname)
        # fpath = args.dir + '/trial_' + str(trial)
        # fname = "init_eigs.P"
        # with open(fpath + fname, 'wb') as fp:
        #     pickle.dump(init_eigs, fp)

        # fname = "final_eigs.P"
        # with open(fpath + fname, 'wb') as fp:
        #     pickle.dump(final_eigs, fp)

def train_epoch(model, loaders, criterion, optimizer, epoch, end_epoch,
            eval_freq = 1, save_freq = 10, output_dir='./', lr_init=0.01):

    time_ep = time.time()

    lr = training_utils.schedule(epoch, lr_init, end_epoch, swa=False)
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

"""
    script to compute maximum and minimum eigenvalues of the hessian
"""
import argparse
import torch

import numpy as np

from hess import data
import hess.nets as models

from hess_vec_prod import min_max_hessian_eigs

parser = argparse.ArgumentParser(description="Eigenvalue calculation")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/datasets/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of train set (default: False)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default="PreResNet56",
    metavar="MODEL",
    help="model name (default: PreResNet56)",
)
parser.add_argument(
    "--save_path", type=str, default=None, required=True, help="path to npz results file"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--nsteps", type=int, default=100, help="number of Lanczos steps (default: 100)"
)
parser.add_argument(
    "--num_channels", type=int, default=64, help="number of channels for resnet"
)
parser.add_argument(
    "--depth", type=int, default=3, help="depth of convnet"
)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

# only use testing data augmentation (e.g. scaling etc.)
# no random flipping
print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_test,
    model_cfg.transform_test,
    use_validation=False,
    split_classes=args.split_classes,
    shuffle_train=False,
)

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                        c=args.num_channels, max_depth=args.depth)
model.cuda()

print("Loading model %s" % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint["state_dict"])

criterion = torch.nn.CrossEntropyLoss()

if args.use_test:
    loader = loaders["test"]
else:
    loader = loaders["train"]

print("computing eigenvalues of the hessian")
min_max_fn = min_max_hessian_eigs
kwargs = {"nsteps": args.nsteps}

# TODO: change this to hess.utils
max_eval, min_eval, hvps, pos_evals, neg_evals, pos_bases = min_max_fn(
    model, loader, criterion, use_cuda=True, verbose=True, **kwargs
)

if neg_evals is not None:
    neg_evals = neg_evals.cpu().numpy()

print("Maximum eigenvalue: ", max_eval)
print("Minimum eigenvalue: ", min_eval)
print("Number of full batch vector products: ", hvps)

print("Saving all eigenvalues to ", args.save_path)
np.savez(
    args.save_path,
    pos_evals=pos_evals.cpu().numpy(),
    pos_bases=pos_bases.cpu().numpy(),
)

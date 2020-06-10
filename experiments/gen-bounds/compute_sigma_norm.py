import math
import torch
import torchvision
import hess
from hess.nets import ConvNetDepth
import torchvision
from torchvision import transforms
from norms import perturb_model, compute_accuracy, mag_sharpness_sigma

def main():


    use_cuda = torch.cuda.is_available()


    ## load in a loader just for sizing ##
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    data_dir = '~/datasets/cifar100/'
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    input_size = next(iter(trainloader))[0].shape

    ## model sizes ##
    depths = torch.arange(9)
    widths = torch.arange(4, 65, 4)

    num_classes = 100

    ## saving ##
    sigma_norms = torch.zeros(depths.numel(), widths.numel())

    for d_ind, dpth in enumerate(depths):
        for w_ind, wdth in enumerate(widths):
            depth = dpth.item()
            width = wdth.item()

            print("depth ", depth, " width ", width, " starting")
            model = ConvNetDepth(num_classes=num_classes, c=width, max_depth=depth)
            if use_cuda:
                model = model.cuda()

            ## YOU'LL NEED TO CHANGE THIS FILE PATHING
            fpath = '../eigenvalues/width_depth_exp/'
            fname = "depth_" + str(depth) + "_width_" + str(width) + "/trial_0/checkpoint-200.pt"

            chckpt = torch.load(fpath + fname)
            model.load_state_dict(chckpt['state_dict'])

            sigma_norms[d_ind, w_ind] = mag_sharpness_sigma(model, trainloader, n_batch_samples=20,
                                                resample_sigma=20, discrep_eps=1e-5, bound_eps=1e-4,
                                                n_midpt_rds=30, upper=0.5, use_cuda=use_cuda)

            print("depth ", depth, " width ", width, " done\n")

        torch.save(sigma_norms, "./width_depth_mag_sigma_norms.pt")


if __name__ == '__main__':
    main()

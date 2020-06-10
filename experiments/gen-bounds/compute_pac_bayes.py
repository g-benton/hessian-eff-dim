import math
import torch
import torchvision
import hess
from hess.nets import PreActBlock, PreActResNet
import torchvision
from torchvision import transforms
from norms import lp_path_norm
from hess import utils

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

    ## model sizes ##
    widths = torch.arange(1, 66, 1)
    num_classes = 100

    ## saving ##
    weight_norms = torch.zeros(widths.numel())
    n_pars = torch.zeros(widths.numel())
    for w_ind, wdth in enumerate(widths):
        width = wdth.item()

        print("width ", width, " starting")
        model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes,
                            init_channels=width)
        if use_cuda:
            model = model.cuda()

        fpath = '~/data/resnet_training_data/'
        fpath2 = "resnet_" + str(width)
        fname = "/trial_0/checkpoint-200.pt"

        chckpt = torch.load(fpath + fpath2 + fname)
        model.load_state_dict(chckpt['state_dict'])
        pars = utils.flatten(model.parameters())
        weight_norms[w_ind] = pars.norm()
        n_pars[w_ind] = sum([p.numel() for p in model.parameters()])

        print("width ", width, " done \n")

    torch.save(weight_norms, "./resnet_weight_norms.pt")
    torch.save(n_pars, "./resnet_n_pars.pt")


if __name__ == '__main__':
    main()

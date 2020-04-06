import math
import torch
import torchvision
import hess
from hess.nets import make_resnet18k
import torchvision
from torchvision import transforms
from norms import lp_path_norm

def main():

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


    trainset = torchvision.datasets.CIFAR10(root='~/datasets/cifar10/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    input_size = next(iter(trainloader))[0].shape

    ## model sizes ##
    depths = torch.arange(9)
    widths = torch.arange(4, 65, 4)

    ## saving ##
    path_norms = torch.zeros(depths.numel(), widths.numel())

    for w_ind, wdth in enumerate(widths):

        print(" width ", width, " starting")
        model = make_resnet18k(k=width, num_classes=100)

        fpath = '/misc/vlgscratch4/WilsonGroup/greg_b/data/resnet_training_data/'
        fpath2 = "resnet_" + str(width)
        fname = "/trial_0/checkpoint-200.pt"

        chckpt = torch.load(fpath + fpath2 + fname, map_location='cpu')
        model.load_state_dict(chckpt['state_dict'])

        path_norms[d_ind, w_ind] = lp_path_norm(model, 'cpu',
                                                input_size=input_size)

    torch.save(path_norms, "./resnet_path_norms.pt")


if __name__ == '__main__':
    main()

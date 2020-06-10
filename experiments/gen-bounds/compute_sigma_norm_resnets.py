import math
import torch
import torchvision
import hess
from hess.nets import PreActBlock, PreActResNet
import torchvision
from torchvision import transforms
from norms import sharpness_sigma

def main():


    use_cuda = torch.cuda.is_available()
    print("use cuda = ", use_cuda)


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
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    input_size = next(iter(trainloader))[0].shape
    ## model sizes ##
    widths = torch.arange(1, 66, 1)
    num_classes = 100

    ## saving ##
    sigma_norms = torch.zeros(widths.numel())
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

        sigma_norms[w_ind] = sharpness_sigma(model, trainloader, target_deviate=0.01, n_batch_samples=10,
                                            resample_sigma=10, discrep_eps=1e-4, bound_eps=1e-4,
                                            n_midpt_rds=20, upper=0.5,
                                            use_cuda=use_cuda)

        print("width ", width, " done \n")

    torch.save(sigma_norms, "./resnet_sigma_norms_new.pt")


if __name__ == '__main__':
    main()

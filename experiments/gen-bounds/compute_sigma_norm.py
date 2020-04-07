import math
import torch
import torchvision
import hess
from hess.nets import PreActBlock, PreActResNet
import torchvision
from torchvision import transforms
from norms import lp_path_norm

def main():


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


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

    data_dir = '/misc/vlgscratch4/WilsonGroup/greg_b/datasets/cifar100/'
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    input_size = next(iter(trainloader))[0].shape

    ## model sizes ##
    widths = torch.arange(1, 66, 1)
    num_classes = 100

    ## saving ##
    path_norms = torch.zeros(widths.numel())
    for w_ind, wdth in enumerate(widths):
        width = wdth.item()

        print("width ", width, " starting")
        model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes,
                            init_channels=width)
        if use_cuda:
            model = model.cuda()

        fpath = '/misc/vlgscratch4/WilsonGroup/greg_b/data/resnet_training_data/'
        fpath2 = "resnet_" + str(width)
        fname = "/trial_0/checkpoint-200.pt"

        chckpt = torch.load(fpath + fpath2 + fname)
        model.load_state_dict(chckpt['state_dict'])

        path_norms[w_ind] = sharpness_sigma(model, trainloader,
                                            use_cuda=use_cuda)

        print("width ", width, " done \n")

    torch.save(path_norms, "./path_norms.pt")


if __name__ == '__main__':
    main()

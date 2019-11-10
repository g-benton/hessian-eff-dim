import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import sys
from hess.nets.cifar_net import cifar_net

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help="number of epochs", default=10, type=int)
    parser.add_argument('--gpu', help='which gpu to run on', default=6, type=int)
    return parser.parse_args()

def main(argv):
    args = parse()
    epochs = args.epochs
    cuda_ = torch.cuda.is_available()
    if cuda_:
        torch.cuda.set_device(args.gpu)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## set up the network ##
    net = cifar_net(num_classes=len(classes), k=128)
    if cuda_:
        net = net.cuda()
        print("net on cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if cuda_:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = '../saved-models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    return 1

if __name__ == '__main__':
    main(sys.argv[1:])

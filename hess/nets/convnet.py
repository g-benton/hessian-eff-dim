## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# based on https://gitlab.com/harvard-machine-learning/double-descent/blob/master/models/mcnn.py

from torchvision import transforms

import torch.nn as nn

def block(input, output):
    # Layer i
    list = [nn.Conv2d(input, output, kernel_size=3,
                stride=1, padding=1, bias=True),
        nn.BatchNorm2d(output),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ]
    return list

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class ConvNetDepth(nn.Module):
    def __init__(self, init_channels=64, num_classes=10, max_depth=3):
        super(ConvNetDepth, self).__init__()
        c = init_channels
        module_list = block(3, c)
        module_list = module_list[:-1] #no max pooling at end of first layer

        current_width = c
        last_zero = max_depth // 3 + 1 * (max_depth%3 > 0) - 1
        for i in range(max_depth // 3 + 1 * (max_depth%3 > 0)):
            if i != last_zero:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1] # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
                current_width = 2 * current_width

        last_one = max_depth // 3 + 1 * (max_depth%3 > 1) - 1
        for i in range(max_depth // 3 + 1 * (max_depth%3 > 1)):
            if i != last_one:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1] # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
                current_width = 2 * current_width

        last_two = max_depth // 3 + 1 * (max_depth%3 > 2) - 1
        for i in range(max_depth // 3 + 1 * (max_depth%3 > 2)):
            if i != last_two:
                module_list.extend(block(current_width, current_width))
                module_list = module_list[:-1] # no max pooling if we repeat layers
            else:
                module_list.extend(block(current_width, 2 * current_width))
                current_width = 2 * current_width

        pooling_increaser = 1
        if max_depth < 3:
            pooling_increaser = (3 - max_depth) * 2

        linear_layer = [
            nn.MaxPool2d(4 * pooling_increaser),
            Flatten(),
            nn.Linear(current_width, num_classes, bias=True)
        ]

        module_list.extend(linear_layer)

        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module_list(x)

class ConvNet:
    base = ConvNetDepth
    args = []
    kwargs = {}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

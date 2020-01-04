"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms
from .masked_layer import MaskedLinear, MaskedConv2d

__all__ = ["VGG16", "VGG16BN", "VGG19", "VGG19BN"]


def make_layers(cfg, batch_norm=False, use_masked=False):
    if use_masked:
        conv_layer = MaskedConv2d
    else:
        conv_layer = nn.Conv2d

    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, use_masked=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm, use_masked=use_masked)
        if use_masked:
            linear_layers = MaskedLinear
        else:
            linear_layers = nn.Linear

        self.classifier = nn.Sequential(
            nn.Dropout(),
            linear_layers(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            linear_layers(512, 512),
            nn.ReLU(True),
            linear_layers(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
        ]
    )


class VGG16(Base):
    pass


class VGG16BN(Base):
    kwargs = {"batch_norm": True}


class VGG19(Base):
    kwargs = {"depth": 19}


class VGG19BN(Base):
    kwargs = {"depth": 19, "batch_norm": True}

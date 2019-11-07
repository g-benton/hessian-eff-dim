import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
class cifar_net(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=128):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            ConvBNrelu(3,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            #nn.Dropout2d(),
            ConvBNrelu(2*k,2*k),
            # ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            #nn.Dropout2d(),
            ConvBNrelu(2*k,2*k),
            # ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

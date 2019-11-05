import math
import torch
from torch import nn
from .simple_net import SimpleNet

class Transformer(nn.Module):
    """docstring for Transformer."""

    def __init__(self, x, y, net=SimpleNet, **kwargs):
        super(Transformer, self).__init__()
        self.x = x
        self.y = y
        self.net = net(x, y, **kwargs)


    def forward(self, x):
        return self.net(x)

    def train_net(self, loss_func=nn.MSELoss(), optim=torch.optim.Adam,
                 lr=0.01, iters=1000, print_loss=False):

        optimizer=optim(self.net.parameters(), lr=lr)

        for epoch in range(iters):
            optimizer.zero_grad()
            outputs = self.net(self.x)

            loss=loss_func(outputs,self.y)
            if print_loss:
                print(loss)
            loss.backward()
            optimizer.step()

        self.net.eval();

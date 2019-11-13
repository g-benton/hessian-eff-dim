import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(train_x, train_y, classifier, use_cuda=False,
                           buffer=0.5, h=0.1):
    x_min, x_max = train_x[:, 0].min() - buffer, train_x[:, 0].max() + buffer
    y_min, y_max = train_x[:, 1].min() - buffer, train_x[:, 1].max() + buffer

    xx,yy=np.meshgrid(np.arange(x_min.cpu(), x_max.cpu(), h), 
                      np.arange(y_min.cpu(), y_max.cpu(), h))
    in_grid = torch.FloatTensor([xx.ravel(), yy.ravel()]).t()
    if use_cuda:
        in_grid = in_grid.cuda()
    Z = classifier(in_grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.detach().cpu(), colors=['red', 'blue'], alpha=0.5,
                levels=1)
    plt.scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), c=train_y[:, 0].cpu(), cmap=plt.cm.binary)
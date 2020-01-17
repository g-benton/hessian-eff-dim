import pyro
import torch
import numpy as np

import pyro.distributions as dist

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = np.linspace(-1, 1, N)
    X = np.power(X[:, np.newaxis], np.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = np.dot(X, W) + 0.5 * np.power(0.5 + X[:, 1], 2.0) * np.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = np.linspace(-1.3, 1.3, N_test)
    X_test = np.power(X_test[:, np.newaxis], np.arange(D_X))

    return X, Y, X_test

def model(X, Y, D_H=15):
    
    D_X, D_Y = X.shape[1], 1
    # sample first layer (we put unit normal priors on all weights)
    w1 = pyro.sample("w1", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H))))  # D_X D_H
    z1 = torch.tanh(torch.matmul(X, w1))   # N D_H  <= first layer of activations

    # sample second layer
    w2 = pyro.sample("w2", dist.Normal(torch.zeros((D_H, D_H)), torch.ones((D_H, D_H))))  # D_H D_H
    z2 = torch.tanh(torch.matmul(z1, w2))  # N D_H  <= second layer of activations

    # sample final layer of weights and neural network output
    w3 = pyro.sample("w3", dist.Normal(torch.zeros((D_H, D_Y)), torch.ones((D_H, D_Y))))  # D_H D_Y
    z3 = torch.matmul(z2, w3)  # N D_Y  <= output of the neural network

    # we put a prior on the observation noise
    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    return pyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)

nuts_kernel = pyro.infer.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples = 100, warmup_steps = 100)

train_x, train_y, test_x = get_data()
train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()

mcmc.run(train_x.cuda(), train_y.cuda())
#hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

from pyro.infer import Predictive
predictive_call = Predictive(model, mcmc.get_samples(), num_samples=100)
pred_samples = predictive_call(test_x.cuda(), None)

import matplotlib.pyplot as plt

plt.plot(test_x, torch.mean(pred_samples,0).detach().cpu().numpy())
plt.scatter(train_x.cpu(), train_y.cpu())
plt.show()
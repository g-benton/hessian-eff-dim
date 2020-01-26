import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt

import pyro.distributions as dist

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def twospirals(n_points, noise=.5, random_state=920):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

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
    # prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    # sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    #return pyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)
    return pyro.sample("Y", dist.Bernoulli(logits=z3), obs=Y)

nuts_kernel = pyro.infer.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples = 1000, warmup_steps = 1000)

train_x, train_y = twospirals(500)
#train_x, train_y, test_x = get_data()
train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
X_test0, X_test1 = np.meshgrid(np.linspace(train_x[:,0].min(), train_x[:,0].max(), 50),
                                   np.linspace(train_x[:,1].min(), train_x[:,1].max(), 50))
test_x = torch.from_numpy(np.vstack((X_test0.ravel(), X_test1.ravel())).T)

mcmc.run(train_x.cuda(), train_y.cuda())

#hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
from pyro.infer import Predictive
predictive_call = Predictive(model, mcmc.get_samples(), num_samples=100)
pred_samples = predictive_call(test_x.float().cuda(), None)

mean_prediction = torch.mean(pred_samples['Y'], 0).detach().cpu().numpy()
plt.contourf(X_test0, X_test1, mean_prediction.reshape(50,50), alpha=0.5)
#plt.colorbar()
plt.scatter(train_x[:,0], train_x[:,1], c=train_y.numpy(), cmap=plt.cm.binary)

#plt.savefig('bnn_plot.pdf')
plt.tight_layout()
plt.show()

# plt.scatter(train_x[:,1].view(-1).cpu(), train_y.view(-1).cpu())
# plt.plot(test_x[:,1], torch.mean(pred_samples['Y'],0).detach().cpu().numpy())

# plt.show()
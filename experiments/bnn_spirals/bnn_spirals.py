# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Bayesian Neural Network
=======================

We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

from jax import vmap
import jax.numpy as np
import jax.random as random
from jax.nn import elu, relu

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

#matplotlib.use('Agg')  # noqa: E402


# the non-linearity we use in our neural network
def nonlin(x):
    #return np.tanh(x)
    return elu(x)

def twospirals(n_points, noise=.5, random_state=920):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(onp.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + onp.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + onp.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H):

    D_X, D_Y = X.shape[1], 1

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))  # D_X D_H
    z1 = nonlin(np.matmul(X, w1))   # N D_H  <= first layer of activations

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(np.zeros((D_H, D_H)), np.ones((D_H, D_H))))  # D_H D_H
    z2 = nonlin(np.matmul(z1, w2))  # N D_H  <= second layer of activations

    w3 = numpyro.sample("w3", dist.Normal(np.zeros((D_H,D_H)), np.ones((D_H,D_H))))
    z3 = nonlin(np.matmul(z2, w3))

    # sample final layer of weights and neural network output
    w4 = numpyro.sample("w4", dist.Normal(np.zeros((D_H, D_Y)), np.ones((D_H, D_Y))))  # D_H D_Y
    z4 = np.matmul(z3, w4)  # N D_Y  <= output of the neural network

    # we put a prior on the observation noise
    #prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    #sigma_obs = 1.0 / np.sqrt(prec_obs)

    # observe data
    #numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)
    numpyro.sample("Y", dist.Bernoulli(logits=z4), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace['Y']['value']

def main(args):
    N, D_H = args.num_data, args.num_hidden
    X, Y = twospirals(n_points=N, noise=0.1)

    X_test0, X_test1 = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 50),
                                   np.linspace(X[:,1].min(), X[:,1].max(), 50))
    X_test = np.vstack((X_test0.ravel(), X_test1.ravel())).T

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, D_H)

    # predict Y_test at inputs X_test
    vmap_args = (samples, random.split(rng_key_predict, args.num_samples * args.num_chains))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H))(*vmap_args)
    predictions = predictions[..., 0]

    # compute mean prediction and confidence interval around median
    mean_prediction = np.mean(predictions, axis=0)
    print(mean_prediction)
    print(mean_prediction.shape, predictions.shape)
    percentiles = onp.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    fig, ax = plt.subplots(1, 1)

    # # plot training data
    # ax.plot(X[:, 1], Y[:, 0], 'kx')
    # # plot 90% confidence level of predictions
    # ax.fill_between(X_test[:, 1], percentiles[0, :], percentiles[1, :], color='lightblue')
    # # plot mean prediction
    # ax.plot(X_test[:, 1], mean_prediction, 'blue', ls='solid', lw=2.0)
    # ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
    ax.contourf(X_test0, X_test1, mean_prediction.reshape(50,50), alpha=0.5)
    #plt.colorbar()
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.binary)

    #plt.savefig('bnn_plot.pdf')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.2.3')
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--num-data", nargs='?', default=100, type=int)
    parser.add_argument("--num-hidden", nargs='?', default=5, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)

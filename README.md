## Rethinking Parameter Counting: Effective Dimensionality Revisted

This repo contains the code needed to replicate the experiments in **Rethinking Parameter Counting: Effective Dimensionality Revisted** by [Wesley Maddox](https://wjmaddox.github.io), [Gregory Benton](https://g-benton.github.io/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

Please cite our work if you find it useful:
```
placeholder for bibtex
```

### Introduction

In this paper we examine the effective dimensionality of the Hessian of the loss (shortened to effective dimensionality) as a way to explain generalization performance in neural networks. We revisit an interpretation offered by McKay (1992): the effective dimensionalilty describes the number of parameters determined by the data. Using this interpretation we find that across many architectures of varying sizes the effective dimensionality of the model provides a much better proxy for generalization than simply counting the number of parameters a model contains.

```
insert first plot from paper
```

```
insert second plot from paper
```

#### Posterior Contraction Experiments; Section 4

To produce the Figure 4 in the paper you will first need to generate the eigenvalues of the Hessian associated with Bayesian neural networks trained using increasing numbers of data points by running `/notebooks/bnn_regression_varying_dimensionality.ipynb`. You will then be able to run `/notebooks/BLR_effective_dimension.ipynb` to compare the effective dimensionality of a Bayesian linear regression with an increasing number of data points.

#### Loss Surface Experiments; Section 5

To see all results from Section 5 of the paper just run `/notebooks/two_spirals.ipynb`. We provide similar results to those of Section 5 for the CIFAR10 dataset. All code needed to generate these results is contained in `experiments/cifar-homogeneity/` and `experiments/cifar-loss-surfaces/`, there are README files in those directories that walk through producing the relevant data and figures.

### Package

To install the package, run `python setup.py develop`. See dependencies in `requirements.txt`. You will need the latest version of PyTorch (>=1.0.0), and standard scipy/numpy builds. Most of the codebase is written to use a GPU if it finds one.


### References
- MacKay, David JC. "Bayesian model comparison and backprop nets." Advances in neural information processing systems. 1992.

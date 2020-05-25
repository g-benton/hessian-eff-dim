## Rethinking Parameter Counting: Effective Dimensionality Revisted

This repo contains the code needed to replicate the experiments in **Rethinking Parameter Counting: Effective Dimensionality Revisted** by [Wesley Maddox](https://wjmaddox.github.io), [Gregory Benton](https://g-benton.github.io/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

Please cite our work if you find it useful:
```
@article{parametercounting,
  title={Rethinking Parameter Counting: Effective Dimensionality Revisted},
  author={Maddox, Wesley J. and Benton, Gregory and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2003.02139},
  year={2020}
}
```

### Introduction

In this paper we examine the effective dimensionality of the Hessian of the loss (shortened to effective dimensionality) as a way to explain generalization performance in neural networks. We revisit an interpretation offered by MacKay (1992): the effective dimensionalilty describes the number of parameters determined by the data. Using this interpretation we find that across many architectures of varying sizes the effective dimensionality of the model provides a much better proxy for generalization than simply counting the number of parameters a model contains.

![Effective Dimensionality and Generalization](plots/dnn_double_descent.png?raw=true "Effective Dimensionality and Generalization")

A resolution of double descent. We replicate the double descent behaviour of deep neural networks using a ResNet18 on CIFAR-100, where train loss decreases to zero with sufficiently wide model while test loss decreases, then increases, and then decreases again. Unlike model width, the effective dimensionality computed from the eigenspectrum of the Hessian of the loss on training data alone follows the test loss in the overparameterized regime, acting as a much better proxy for generalization than naive parameter counting.

![Effective Dimensionality, Width, and Depth](plots/width-depth-exp.png?raw=true "Effective Dimensionality, Width, and Depth")
**Left**: Effective dimensionality as a function of model width and depth for a CNN on CIFAR100. **Center**: Test loss as a function of model width and depth. **Right**: Train loss as a function of model width and depth. Yellow level curves represent equal parameter counts (1e5, 2e5, 4e5, 1.6e6). The green curve separates models with near-zero training loss. Effective dimensionality serves as a good proxy for generalization for models with low train loss. We see wide but shallow models overfit, providing low train loss, but high test loss and high effective dimensionality. For models with the same train loss, lower effective dimensionality can be viewed as a better compression of the data at the same fidelity. Thus depth provides a mechanism for compression, which leads to better generalization.

### Package

To install the package, run `python setup.py develop`. See dependencies in `requirements.txt`. You will need the latest version of PyTorch (>=1.0.0), and standard scipy/numpy builds. Most of the codebase is written to use a GPU if it finds one.

#### Computing Effective Dimensionality

To compute the effective dimensionality of a network you need only compute the dominant eigenvalues of the Hessian of the loss of the network. We provide code that will manage the eigenvalue computation for you using the Lanczos algorithm in `/experiments/eigenvalues/run_hess_eigs.py`. An example call to this script is as follows:

```bash
python run_hess_eigs.py --dataset=CIFAR100 --data_path=/path/to/dataset/ --model=ResNet18 \
        --num_channels=$channels --file=your_saved_model.pt \
        --save_path=your_output_eigenvalues.npz
```

#### Double Descent Experiments; Figures 1 and 2

To reproduce Figures 1 and 2 in the paper you will need to train a number of models and compute their effective dimensionalities. The files and command line instructions to produce all results necessary to recreate both figures are in `/experiments/eigenvalues/`.

#### Posterior Contraction Experiments; Section 4

To produce the Figure 4 in the paper you will first need to generate the eigenvalues of the Hessian associated with Bayesian neural networks trained using increasing numbers of data points by running `/notebooks/bnn_regression_varying_dimensionality.ipynb`. You will then be able to run `/notebooks/BLR_effective_dimension.ipynb` to compare the effective dimensionality of a Bayesian linear regression with an increasing number of data points.

#### Loss Surface Experiments; Section 5

To see all results from Section 5 of the paper just run `/notebooks/two_spirals.ipynb`. We provide similar results to those of Section 5 for the CIFAR10 dataset. All code needed to generate these results is contained in `experiments/cifar-homogeneity/` and `experiments/cifar-loss-surfaces/`, there are README files in those directories that walk through producing the relevant data and figures.

### References
- MacKay, David JC. "Bayesian model comparison and backprop nets." Advances in neural information processing systems. 1992.
- Maddox et al. "Rethinking Parameter Counting in Deep Models: Effective Dimensionality Revisited." 2020 (https://arxiv.org/abs/2003.02139)

## References for Code Base

Model implementations:
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - ResNet18: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py, but copied from https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/resnet18k.py
  - CNNs: copied from https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/mcnn.py 
  

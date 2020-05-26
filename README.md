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

In this paper we examine the effective dimensionality of the Hessian of the loss (shortened to effective dimensionality) as a way to explain generalization performance in neural networks. Effective dimensionality is computed as,

<img src="https://render.githubusercontent.com/render/math?math=N_{eff}(\textrm{Hessian})=\sum_{i=1}^{j}\frac{\lambda_j}{\lambda_j%2B\alpha},">

where <img src="https://render.githubusercontent.com/render/math?math=\lambda_j"> are the eigenvalues of the Hessian at the converged solution on the training loss and <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is a regularization term.
We revisit an interpretation offered by MacKay (1992): the effective dimensionalilty describes the number of parameters determined by the data. 
Using this interpretation we find that across many architectures of varying sizes the effective dimensionality of the model provides a much better proxy for generalization than simply counting the number of parameters a model contains.

![Effective Dimensionality and Generalization](plots/dnn_double_descent.png?raw=true "Effective Dimensionality and Generalization")

A resolution of double descent. We replicate the double descent behaviour of deep neural networks using a ResNet18 on CIFAR-100, where train loss decreases to zero with sufficiently wide model while test loss decreases, then increases, and then decreases again. Unlike model width, the effective dimensionality computed from the eigenspectrum of the Hessian of the loss on training data alone follows the test loss in the overparameterized regime, acting as a much better proxy for generalization than naive parameter counting.

![Effective Dimensionality, Width, and Depth](plots/width-depth-exp.png?raw=true "Effective Dimensionality, Width, and Depth")
**Left**: Effective dimensionality as a function of model width and depth for a CNN on CIFAR100. **Center**: Test loss as a function of model width and depth. **Right**: Train loss as a function of model width and depth. Yellow level curves represent equal parameter counts (1e5, 2e5, 4e5, 1.6e6). The green curve separates models with near-zero training loss. Effective dimensionality serves as a good proxy for generalization for models with low train loss. We see wide but shallow models overfit, providing low train loss, but high test loss and high effective dimensionality. For models with the same train loss, lower effective dimensionality can be viewed as a better compression of the data at the same fidelity. Thus depth provides a mechanism for compression, which leads to better generalization.

### Package

To install the package, run `python setup.py develop`. See dependencies in `requirements.txt`. You will need the latest version of PyTorch (>=1.0.0), and standard scipy/numpy builds. Most of the codebase is written to use a GPU if it finds one.

#### Computing Effective Dimensionality

To compute the effective dimensionality of a network you only need to compute the dominant eigenvalues of the Hessian of the loss of the network, and make a choice of the regularization term <img src="https://render.githubusercontent.com/render/math?math=\alpha">. 

In the paper that for Bayesian Linear models the effective dimensionality corresponds the contraction of the posterior distribution when <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is chosen to be the product of the number of datapoints _N_ and the prior variance. For neural networks an alternate choice of regularization constant is then  _N * (weight decay)_, which is a reasonable heuristic for <img src="https://render.githubusercontent.com/render/math?math=\alpha">. In general, the qualitative behaviour of effective dimensionality is quite robust to the value of <img src="https://render.githubusercontent.com/render/math?math=\alpha"> above a certain threshold (a value of 1 is reasonable in the double descent and width-depth experiments). Often the eigenspectrum of the Hessian contains a kink that separates a small number of relatively larger eigenvalues from the rest. One would want to choose an <img src="https://render.githubusercontent.com/render/math?math=\alpha"> larger than the smallest "large" eigenvalue, so that effective dimension is measuring the number of significantly determined directions.

We provide code that will manage the eigenvalue computation for you using the Lanczos algorithm in `/experiments/eigenvalues/run_hess_eigs.py`. An example call to this script is as follows:

```bash
python run_hess_eigs.py --dataset=CIFAR100 --data_path=/path/to/dataset/ --model=ResNet18 \
        --num_channels=$channels --file=your_saved_model.pt \
        --save_path=your_output_eigenvalues.npz
```

If you just wish to use the codebase to compute effective dimensionality of a neural network, the easiest way to proceed is to modify the `experiments/eigenvalues/run_hess_eigs.py` script to allow for loading and computation of your specific model on a dataset.
You will then be able to use the command line args to compute as many eigenvalues as you want and output them into a npz file.
Then, 
```python
import numpy as np
eigs = np.load('path.npz')['pos_evals']

def eff_dim(x, s = 1.):
    x = x[x!=1.] #remove eigenvalues that didnt converge from the lanczos computation to make things less noisy
    return np.sum(x / (x + s))
    
eff_dim(eigs)
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
  

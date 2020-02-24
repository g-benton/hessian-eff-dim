## To produce the plots in figure A.2
There is a trained model already in this directory
1. Run `train_cifar10.py` to produce a trained model 
2. Run `get_evals.py`to compute the top 200 eigenvalue eigenvector pairs of the Hessian
3. Run `perturbations.py` to compute the loss and prediction differences under different perturbations to the parameters
4. Explore the output with the notebook `homogeneity_plotter.ipynb`.

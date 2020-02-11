## To produce the plots in figure A.2
There is a trained model already in this directory
1. Run `get_evals.py`to compute the top 200 eigenvalue eigenvector pairs of the Hessian
2. Run `perturbations.py` to compute the loss and prediction differences under different perturbations to the parameters
3. Explore the output with the notebook `homogeneity_plotter.ipynb`.

import math
import torch
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

def features(x, p=10):
    phi = torch.zeros(x.numel(), p)
    ind = 0
    for freq in range(p//2):
        phi[:, ind] = torch.cos((freq+1)*math.pi*x)
        ind += 1
        phi[:, ind] = torch.sin((freq+1)*math.pi*x)
        ind += 1

    return phi

class BLRModel(torch.nn.Module):
    """
    just a dummy class to evaluate \beta to be the mean of the posterior
    """
    def __init__(self, fmap=features, n_features=200, prior_var=1.):
        super(BLRModel, self).__init__()
        self.fmap = fmap
        self.n_features = n_features
        self.prior_var = prior_var
        self.register_parameter(
            name="beta", param=torch.nn.Parameter(torch.zeros(n_features,1))
        )

    def update_pars(self, x, y, sig):
        S_0 = torch.eye(self.prior_var).inverse()
        features = self.fmap(x, p=self.n_features)
        in_term = features.t().matmul(features).div(sig**2)
        in_term += torch.eye(self.n_features).mul(self.prior_var).inverse()
        self.beta.data = in_term.inverse().matmul(features.t().matmul(y)).div(sig**2)

def post_cov(phi, sig, s_0):
    in_ = phi.t().matmul(phi).div(sig) + s_0.inverse()
    return in_.inverse()

def log_likelihood(model, x, y, sig):
    mean = model.beta.data
    phi = model.fmap(x, p=model.n_features)
    prior_var = torch.eye(model.n_features).mul(sig)
    cov = post_cov(phi, sig, prior_var)

    post_dist = MultivariateNormal(loc=mean.squeeze(), covariance_matrix=cov)
    return post_dist.log_prob(model.beta.squeeze() + torch.ones_like(model.beta.squeeze())*0.01)

def NormalHessianProd(vec, model, criterion, inputs=None, targets=None,
                        sig=1., use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.
    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
    """
    if use_cuda:
        vec = vec.cuda()

    model.zero_grad()  # clears grad for every parameter in the net
    inputs, targets = Variable(inputs), Variable(targets)

    loss = criterion(model, inputs, targets, sig)
    grad_f = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    # print(grad_f)

    # Compute inner product of gradient with the direction vector
    # prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
    prod = torch.zeros(1, dtype=grad_f[0].dtype, device=grad_f[0].device)
    for (g, v) in zip(grad_f, vec):
        prod = prod + (g * v).sum()

    # Compute the Hessian-vector product, H*v
    # prod.backward() computes dprod/dparams for every parameter in params and
    # accumulate the gradients into the params.grad attributes
    prod.backward()

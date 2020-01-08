import torch
import torch.nn.functional as F
import copy

from ..utils import flatten, unflatten_like

def fisher_trace(inputs, targets, diag_pars, model, base_loss = F.cross_entropy, beta = 0.01, samples = 1, nugget=1e-5):
    model_state_dict = copy.deepcopy(model.state_dict())
    param_vec = flatten(model.parameters()).view(-1,1)
    
    dist = torch.distributions.Normal(torch.zeros_like(param_vec), F.softplus(nugget + diag_pars))
    total_loss = 0.
    for _ in range(samples):
        new_param_vec = dist.rsample(torch.Size((1,)))
        new_param_list = unflatten_like(new_param_vec, model.parameters())
        # set parameters in model
        for p, np in zip(model.parameters(), new_param_list):
            p.detach_()
            p.add_(np)

        total_loss = total_loss + base_loss(model(inputs), targets)

        # for p, np in zip(model.parameters(), new_param_list):
        #     p.add_(-np)

    # for p in model.parameters():
    #     p.requires_grad = True

    avg_loss = total_loss / samples

    model.load_state_dict(model_state_dict)

    return avg_loss + beta * torch.log(F.softplus(nugget + diag_pars)).sum()
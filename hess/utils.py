import torch
import time
import numpy as np
import hess
from torch import nn
from torch.autograd import Variable

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

################################################################################
#                              Supporting Functions
################################################################################
def gradtensor_to_tensor(net, include_bn=False):
    """
        convert the grad tensors to a list
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return flatten([p.grad.data for p in net.parameters() if filter(p)])


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, net, criterion, inputs=None, targets=None,
                       dataloader=None,
                       use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.
    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

#     net.eval()
    net.zero_grad()  # clears grad for every parameter in the net
    if dataloader is None:
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=net.parameters(), create_graph=True)

        # Compute inner product of gradient with the direction vector
        # prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        prod = torch.zeros(1, dtype=grad_f[0].dtype, device=grad_f[0].device)
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).sum()

        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward()
    else:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            #inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            grad_f = torch.autograd.grad(loss, inputs=net.parameters(), create_graph=True)
            # Compute inner product of gradient with the direction vector
            # prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
            #print(grad_f[0])
            #prod = torch.zeros(1, dtype=grad_f[0].dtype, device=grad_f[0].device)
            prod = 0.
            for (g, v) in zip(grad_f, vec):
                prod = prod + (g * v).sum()

            # Compute the Hessian-vector product, H*v
            # prod.backward() computes dprod/dparams for every parameter in params and
            # accumulate the gradients into the params.grad attributes
            prod.backward()

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

#####################################################
# Gets the mask as a torch tensor from a masked net #
#####################################################
def get_mask(net):
    mask_list = []
    for lyr in net.modules():
        if isinstance(lyr, hess.nets.MaskedLinear) or isinstance(lyr, hess.nets.MaskedConv2d):
            mask_list.append(lyr.mask)
            if lyr.has_bias:
                mask_list.append(lyr.bias_mask)
        elif isinstance(lyr, nn.Linear) or isinstance(lyr, nn.Conv2d):
            for p in lyr.parameters():
                mask_list.append(torch.ones_like(p.data))
        elif isinstance(lyr, nn.BatchNorm2d):
            mask_list.append(torch.ones_like(lyr.weight))
            mask_list.append(torch.ones_like(lyr.bias))
            #print(list(lyr.named_parameters()))

    return flatten(mask_list)

def apply_mask(model, mask):
    mask_ind = 0
    for lyr in model.modules():
        if hasattr(lyr, "mask"):
            lyr.mask = mask[mask_ind]
            mask_ind += 1

    return


#############################
# Return Hessian of a model #
#############################

def get_hessian(train_x, train_y, loss, model, use_cuda=False):
    n_par = sum(torch.numel(p) for p in model.parameters())
    hessian = torch.zeros(n_par, n_par)
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    for pp in range(n_par):
        base_vec = torch.zeros(n_par).unsqueeze(0)
        if use_cuda:
            base_vec = base_vec.cuda()
            model = model.cuda()

        base_vec[0, pp] = 1.

        base_vec = unflatten_like(base_vec, model.parameters())
        eval_hess_vec_prod(base_vec,
                                # model.parameters(),
                                net=model,
                                criterion=loss,
                                inputs=train_x, targets=train_y)
        if pp == 0:
            output = gradtensor_to_tensor(model, include_bn=True)
            hessian = torch.zeros(output.nelement(), output.nelement())
            hessian[:, pp] = output

        hessian[:, pp] = gradtensor_to_tensor(model, include_bn=True)

    return hessian


def get_hessian_eigs(loss, model, mask,
                     use_cuda=False, n_eigs=100, train_x=None, train_y=None,
                     loader=None, evals=False):
    if train_x is not None:
        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

    if n_eigs != -1:
        numpars = int(mask.sum().item())
        total_pars = sum(m.numel() for m in model.parameters())

        def hvp(rhs):
            padded_rhs = torch.zeros(total_pars, rhs.shape[-1],
                                     device=rhs.device, dtype=rhs.dtype)
            padded_rhs[mask==1] = rhs
            padded_rhs = unflatten_like(padded_rhs.t(), model.parameters())
            eval_hess_vec_prod(padded_rhs, net=model,
                               criterion=loss, inputs=train_x,
                               targets=train_y, dataloader=loader, use_cuda=use_cuda)
            full_hvp = gradtensor_to_tensor(model, include_bn=True)
            sliced_hvp = full_hvp[mask==1].unsqueeze(-1)
#             print('finished a hvp')
            return sliced_hvp

#         print('numpars is: ', numpars)
        if train_x is None:
            data = next(iter(loader))[0]
            if use_cuda:
                data = data.cuda()
            dtype = data.dtype
            device = data.device
        else:
            dtype, device = train_x.dtype, train_x.device

        qmat, tmat = lanczos_tridiag(hvp, n_eigs, dtype=dtype,
                                  device=device, matrix_shape=(numpars,
                                  numpars))
        eigs, t_evals = lanczos_tridiag_to_diag(tmat)
        if evals:
            return eigs, qmat @ t_evals
        return eigs
    else:
        # form and extract sub hessian
        hessian = get_hessian(train_x, train_y, loss, model, use_cuda=use_cuda)

        keepers = np.array(np.where(mask.cpu() == 1))[0]
        sub_hess = hessian[np.ix_(keepers, keepers)]
        e_val, _ = np.linalg.eig(sub_hess.cpu().detach())
        return e_val.real


def mask_model(model, pct_keep, use_cuda=False):
    n_par = sum(torch.numel(p) for p in model.parameters())
    n_keep = int(pct_keep * n_par)

    mask = [1 for i in range(n_keep)] + [0 for i in range(n_par - n_keep)]
    mask = torch.tensor(mask)
    perm = np.random.permutation(n_par)
    mask = mask[perm]
    if use_cuda:
        mask = mask.cuda()

    mask = unflatten_like(mask.unsqueeze(0), model.parameters())

    mask_ind = 0
    for lyr in model.sequential:
        if isinstance(lyr, hess.nets.MaskedLinear):
            lyr.mask = mask[mask_ind]
            mask_ind += 1
            if lyr.has_bias:
                lyr.bias_mask = mask[mask_ind]
                mask_ind += 1

    return flatten(mask), perm

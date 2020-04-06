import torch
import math
import copy
import warnings
import torch.nn as nn
import numpy as np
from hess import utils


# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.data = param.data.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p )).item()


def perturb_model(model, sigma, n_pars):
    perturb = torch.randn(n_pars) * sigma
    perturb = utils.unflatten_like(perturb.unsqueeze(0), model.parameters())

    for i, par in enumerate(model.parameters()):
        par.data = par.data + perturb[i]

    return

def compute_error(model, dataloader, n_batch_samples):
    error = 0
    for batch in range(n_batch_samples):
        images, labels = next(iter(dataloader))
        preds = model(images).max(-1)[1]
        error += torch.where(preds != labels)[0].numel()

    return error/(n_batch_samples * dataloader.batch_size)



def sharpness_sigma(model, trainloader, target_deviate=0.1, resample_sigma=10,
                    n_batch_samples=10, n_midpt_rds=15, upper=1., lower=0.,
                    bound_eps=5e-3, discrep_eps=1e-2):

    train_accuracy = compute_error(model, trainloader, n_batch_samples)

    saved_pars = model.state_dict()
    n_pars = sum([p.numel() for p in model.parameters()])

    for midpt_iter in range(n_midpt_rds):
        model.load_state_dict(saved_pars)
        midpt = (upper + lower)/2.

        rnd_errors = torch.zeros(resample_sigma)
        perturb_model(model, midpt, n_pars)
        for rnd in range(resample_sigma):
            rnd_errors[rnd] = compute_error(model, trainloader, n_batch_samples)

        print(rnd_errors)
        rnd_error = rnd_errors.mean()

        discrepancy = torch.abs(train_accuracy - rnd_error)

        if ((upper - lower) < bound_eps) or (discrepancy < discrep_eps):
            return midpt

        elif rnd_error > target_deviate:
            ## can cutoff the upper half
            upper = midpt
            print("cutoff upper\n")
        else:
            ## can cutoff the lower half
            lower = midpt
            print("cut off lower\n")


    return midpt
#
#     nxt_data = next(iter(trainloader))
#     image_tensor = nxt_data[0]
#     label_tensor = nxt_data[1]
#
#     logits = model(image_tensor)
#     saved_pars = model.state_dict().clone()
#
#     perturb_ph, perturb_add, perturb_sub = add_noise_to_variables(model_variables)
#     perturb_ph_list = [perturb_ph[k] for k in perturb_ph]
#     original_weight = [sess.run(v) for v in model_variables]
#     restore_original_weight_op = [
#       tf.assign(v, vv) for v, vv in zip(model_variables, original_weight)
#     ]
#
#     perturbation = []
#     for v, vo in zip(model_variables, original_weight):
#         perturbation.append(v - vo)
#     flattened_difference = flatten_and_concat(perturbation)
#      perturb_norm = tf.norm(flattened_difference)
#
#     projection_op = []
#     target_norm = tf.placeholder(tf.float32, ())
#     for v, vo in zip(model_variables, original_weight):
#         projection_op.append(
#             tf.assign(v, vo + target_norm / perturb_norm * (v - vo)))
#
#     xent = tf.nn.softmax_cross_entropy_with_logits(
#         logits=logits, labels=label_tensor)
#     op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-xent)
#     correctly_predicted = tf.equal(
#       tf.argmax(logits, axis=-1), tf.argmax(label_tensor, axis=-1))
#     loss = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32))
#
#     h, l = upper, lower
#     for j in range(search_depth):
#         m = (h + l) / 2.
#         min_accuracy = 10.
#         for i in range(mtc_iter):
#           sess.run(restore_original_weight_op)
#           fd = get_gaussian_noise_feed_dict(perturb_ph, m)
#           sess.run(perturb_add)
#           for k in range(ascent_step):
#             sess.run(op)
#             new_norm = sess.run(perturb_norm)
#             if new_norm > m: sess.run(projection_op, {target_norm: m})
#             if j % 10 == 0:
#               estimates = []
#               for _ in range(num_batch):
#                 estimates.append(sess.run(loss))
#               min_accuracy = min(min_accuracy, np.mean(estimates))
#         deviate = abs(min_accuracy - training_accuracy)
#         if h - l < bound_eps or abs(deviate - target_loss) < deviat_eps:
#           return m
#         if deviate > target_loss:
#           h = m
#         else:
#           l = m

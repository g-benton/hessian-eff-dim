import numpy as np

epochs = [0, 150, 300]
args = ["fisher", "ntk"]
model = "wrn_c100_"

full_evals_list = []
for arg in args:
    epoch_list = []
    for epoch in epochs:
        npz_arr = np.load(model + arg + "_epoch" + str(epoch) + ".npz")
        pos_evals = npz_arr["pos_evals"]
        # pos_bases = npz_arr['pos_bases']
        epochs_evals = np.zeros_like(pos_evals) + epoch
        arg_evals = [arg] * len(pos_evals)
        epoch_list.append([pos_evals, epochs_evals, arg_evals])
    full_evals_list.append(epoch_list)

np.save(full_evals_list)

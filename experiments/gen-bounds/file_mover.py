import torch
import os
import shutil

def main():
    ## model sizes ##
    depths = torch.arange(9)
    widths = torch.arange(4, 65, 4)

    for d_ind, dpth in enumerate(depths):
        for w_ind, wdth in enumerate(widths):
            depth = dpth.item()
            width = wdth.item()

            print("depth ", depth, " width ", width, " starting")

            fpath = './saved-outputs/width_depth_exp_all/'
            fpath2 = "depth_" + str(depth) + "_width_" + str(width) + "/trial_0/"
            fname = "checkpoint-200.pt"

            old_path = fpath + fpath2 + fname

            new_fpath = "./saved-outputs/width-depth-checkpoints/"
            new_fname = "depth_" + str(depth) + "_width_" + str(width) + "_checkpt.pt"

            new_path = new_fpath + new_fname
            shutil.move(old_path, new_path)




if __name__ == '__main__':
    main()

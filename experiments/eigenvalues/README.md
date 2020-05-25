This folder contains the scripts to generate Figures 1 and 2 in the paper.
Specifically, we include one training script `train.py` and another script for eigenvalue computation of the Hessian
with Lanczos (`run_hess_eigs.py`). We use a slightly modified hessian vector product function (`hess_vec_prod.py`) 
that allows us to return the (approximate) eigenvectors if we want to.

Example bash script for Figure 1. Note that width = 64 is the standard width of a ResNet18.

```bash
for channels in {1..65};
do
        echo $channels
        python train.py --dataset=CIFAR100 --data_path=~/datasets/ --model=ResNet18 --lr=0.01 --wd=1e-4 \
               --momentum=0.9 --dir=resnet_$channels --num_channels=$channels
        python run_hess_eigs.py --dataset=CIFAR100 --data_path=~/datasets/ --model=ResNet18 \
                --num_channels=$channels --file=resnet_$channels/trial_0/checkpoint-200.pt \
                --save_path=hessian_eigs/width_$channels.npz
done
```

Example bash script for Figure 2. Again, note that width = 64 is the standard width of the ConvNets we used.
Note that this will take a very long time to run.
We actually didn't use this training loop because we submitted a bunch of slurm jobs instead.

```bash
for depth in {0..8};
do
       for channels in {4..64..4};
       do
               python train.py --dataset=CIFAR100 --data_path=~/datasets/ --epochs=200 \
                    --lr=0.01 --momentum=0.9 --wd=1e-4  --model=ConvNet --num_channels=$channels \
                    --depth=$depth --save_freq=50 --use_test \
                    --dir=width_depth_exp/depth_$depth_width_$channels
                python run_hess_eigs.py --dataset=CIFAR100 --data_path=~/datasets/ --model=ResNet18 \
                    --num_channels=$channels \
                    --file=width_depth_exp/depth_$depth_width_$channels/trial_0/checkpoint-200.pt \
                    --save_path=width_depth_exp/depth_${depth}_width_$channels.npz
       done
done
```
                    

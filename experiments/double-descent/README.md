This folder contains the scripts to generate Figures 1 and 2 in the paper.
Specifically, we include one training script `train.py` and another script for eigenvalue computation of the Hessian
with Lanczos (`run_hess_eigs.py`). We use a slightly modified hessian vector product function (`hess_vec_prod.py`) 
that allows us to return the (approximate) eigenvectors if we want to.

Example bash script for generating the data for Figure 1. Note that width = 64 is the standard width of a ResNet18. Note that you probably
do not want to run this script on a single GPU as it will take a long time.

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

Example bash script for generating the data for Figure 2. This script will take even longer to run
```bash
depth=$2
echo "running models with depth $2"

for i in {4..64..4};
do
  	echo $depth $i
        mkdir width_depth_exp/depth_${depth}_width_$i
        python train_depth.py --dataset=CIFAR100 --data_path=${expHome}/datasets/ --epochs=$
                --lr=0.01 --momentum=0.9 --wd=1e-4 --model=ConvNet --num_channels=$i \
                --depth=$depth --save_freq=50 --use_test \
                --dir=${expHome}/width_depth_exp/depth_${depth}_width_$i \
                > ${expHome}/width_depth_exp/depth_${depth}_width_$i.log
        python run_hess_eigs.py --dataset=CIFAR100 --data_path=${expHome}/datasets/ --model=$
                --num_channels=$i --depth=$depth \
                --file=${expHome}/width_depth_exp/depth_${depth}_width_$i/trial_0/checkpoint$
                --save_path=${expHome}/width_depth_exp/width_${i}_depth_${depth}.npz

done
```

On our system, we ran these commands by 
```bash
bash eigs_cli.sh 3 #where the 3 stands for the depth of the model.
```

We used a slurm scheduler to submit these commands :).
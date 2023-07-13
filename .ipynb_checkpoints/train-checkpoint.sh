#!/bin/bash

# SLurm sbatch options
#SBATCH --gres=gpu:volta:1
#SBATCH -o log.out
#SBATCH -c 8

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/lib:$LD_LIBRARY_PATH

python3 train.py

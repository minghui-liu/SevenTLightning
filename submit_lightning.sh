#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=32gb
#SBATCH --qos high
#SBATCH -t 1-00:00:00

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=1

module add cuda
module add Python3

# ulimit -n 2048
source /fs/nexus-scratch/minghui/NMSS/MSProject/.venv/bin/activate

srun -u python3 train_lightning.py --model unetr -w 1 --scale_intensity_clip False --epochs 1000 -lr 0.0002 --devices 2 -b 8

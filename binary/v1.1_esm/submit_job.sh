#!/bin/bash
#SBATCH --job-name=arg_binary_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 加载模块（某些集群需要）
# module load cuda/12.1
# module load cudnn/8.9

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arg_binary

cd model_train
export CUDA_VISIBLE_DEVICES=0

python train.py

echo "Job completed at $(date)"

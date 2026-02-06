#!/bin/bash
#SBATCH --job-name=arg_multi_train    # 作业名
#SBATCH --partition=gpu               # GPU分区名（根据你的集群修改）
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks-per-node=1           # 任务数
#SBATCH --cpus-per-task=8             # CPU核数
#SBATCH --mem=64G                     # 内存
#SBATCH --gres=gpu:1                  # 申请1块GPU
#SBATCH --time=12:00:00               # 最大运行时间
#SBATCH --output=logs/slurm_%j.out    # 标准输出
#SBATCH --error=logs/slurm_%j.err     # 错误输出

# 加载模块（某些集群需要）
# module load cuda/12.1
# module load cudnn/8.9

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh  # 根据你的conda安装路径修改
conda activate arg_multi

# 进入工作目录
cd model_train

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 启动训练（使用nohup防止SSH断开）
python train.py

# 或者后台运行+日志
# nohup python train.py > ../logs/train_${SLURM_JOB_ID}.log 2>&1 &
# wait

echo "Job completed at $(date)"

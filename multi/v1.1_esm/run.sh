#!/bin/bash
# 多分类模型训练启动脚本（直接运行，无需 SLURM）

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arg_multi

# 进入工作目录
cd model_train

# 设置 CUDA 设备（单 GPU）
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p ../logs

# 启动训练（后台运行，日志保存）
nohup python train.py > ../logs/train_$(date +%Y%m%d_%H%M).log 2>&1 &

# 打印进程信息
echo "训练已启动，PID: $!"
echo "日志文件: ../logs/train_$(date +%Y%m%d_%H%M).log"
echo ""
echo "查看实时日志:"
echo "  tail -f ../logs/train_$(date +%Y%m%d_%H%M).log"
echo ""
echo "查看 GPU 状态:"
echo "  watch -n 2 nvidia-smi"

#!/bin/bash
# 二分类模型训练启动脚本（直接运行，无需 SLURM）

source ~/miniconda3/etc/profile.d/conda.sh
conda activate arg_binary

cd model_train

export CUDA_VISIBLE_DEVICES=0

mkdir -p ../logs

nohup python train.py > ../logs/train_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "训练已启动，PID: $!"
echo "日志文件: ../logs/train_$(date +%Y%m%d_%H%M).log"
echo ""
echo "查看实时日志:"
echo "  tail -f ../logs/train_$(date +%Y%m%d_%H%M).log"
echo ""
echo "查看 GPU 状态:"
echo "  watch -n 2 nvidia-smi"

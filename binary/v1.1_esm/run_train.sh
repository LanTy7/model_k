#!/bin/bash
# 训练脚本 - ARG 二分类模型 v1.1

cd model_train

echo "======================================"
echo "ARG Binary Classification v1.1 Training"
echo "======================================"
echo ""

# 检查依赖
echo "Checking dependencies..."
python -c "import torch; import transformers; import Bio; print('All dependencies installed!')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing dependencies. Please install:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No GPU detected. Training will be slow on CPU."
fi

echo ""
echo "Starting training..."
python train.py

echo ""
echo "Training completed!"
echo "Check ./well-trained/ for saved models"
echo "Check ./figures/ for training curves"

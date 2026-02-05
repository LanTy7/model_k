# ARG 二分类模型 v1.1 - ESM-2 + BiLSTM

## 改进点

相比 v1.0 版本，v1.1 引入了以下改进：

1. **ESM-2 预训练模型**：使用 Facebook 的 ESM-2 (Evolutionary Scale Modeling) 作为特征提取器
   - 支持多种模型尺寸：esm2_t6_8M, esm2_t12_35M, esm2_t30_150M
   - 可配置冻结层数，平衡性能和训练速度

2. **注意力池化机制**：在 BiLSTM 基础上添加 Multi-head Attention，增强全局特征提取

3. **分层学习率**：ESM-2 使用较小学习率 (LR*0.1)，其他参数使用正常学习率

4. **更稳定的训练策略**：
   - 梯度裁剪 (clip_grad_norm)
   - 改进的早停机制
   - 更细致的学习率调度

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

### 基本用法

```bash
cd model_train
python train.py
```

### 修改配置

编辑 `train.py` 中的 `Config` 类：

```python
class Config:
    # ESM-2 模型选择
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 小模型，快速实验
    # ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  # 中等模型
    # ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"  # 大模型，效果更好
    
    # 训练参数
    MAX_LENGTH = 1000
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 100
    
    # 数据路径（必须修改）
    ARG_FILE = "/path/to/your/arg_sequences.fasta"
    NON_ARG_FILE = "/path/to/your/non_arg_sequences.fasta"  # 可选
```

## 使用模型进行预测

```bash
cd model_test

# 单个文件
python predict.py \
    --model ../well-trained/esm2_bilstm_binary_YYYYMMDD_HHMM.pth \
    --input /path/to/input.fasta \
    --output ./results \
    --threshold 0.5

# 批量处理目录
python predict.py \
    --model ../well-trained/esm2_bilstm_binary_YYYYMMDD_HHMM.pth \
    --input /path/to/input_directory/ \
    --output ./results \
    --threshold 0.5
```

### 预测参数

- `--model`: 训练好的模型路径
- `--input`: 输入 FASTA 文件或目录
- `--output`: 输出目录
- `--threshold`: 分类阈值 (默认 0.5)
- `--device`: 使用 cuda 或 cpu
- `--batch_size`: 批大小（可选，覆盖配置）

## 输出结果

预测结果包括：
1. `predictions.csv` - 包含所有序列的预测概率
2. `{filename}_predicted.fasta` - 预测的 ARG 序列

CSV 格式：
| 列名 | 说明 |
|------|------|
| id | 序列 ID |
| description | 原始描述 |
| sequence | 序列 |
| arg_probability | ARG 概率 |
| is_arg | 是否为 ARG |
| source_file | 源文件 |

## 模型架构

```
Input Sequence
    ↓
ESM-2 Tokenizer (<cls> + seq + <eos>)
    ↓
ESM-2 Model (frozen/unfrozen layers)
    ↓ (batch, seq_len, hidden_size)
BiLSTM (2 layers, bidirectional)
    ↓ (batch, seq_len, hidden*2)
Attention Pooling / Global Pooling
    ↓ (batch, hidden*2 or hidden*4)
Classifier (Linear -> ReLU -> Dropout -> Linear)
    ↓
Output Probability (Sigmoid)
```

## 性能对比

| 模型 | 参数量 | 预计训练时间 | 预计准确率 |
|------|--------|-------------|-----------|
| BiLSTM (v1.0) | ~1M | 30 min | ~85% |
| ESM-2 8M + BiLSTM | ~10M | 2-3 hours | ~90-92% |
| ESM-2 35M + BiLSTM | ~40M | 4-6 hours | ~92-94% |
| ESM-2 150M + BiLSTM | ~160M | 8-12 hours | ~93-95% |

## 注意事项

1. **显存需求**：ESM-2 模型较大，建议使用显存 >= 8GB 的 GPU
2. **批大小**：如果显存不足，请减小 BATCH_SIZE（如 16 或 8）
3. **冻结层数**：初次实验建议冻结 4-6 层 ESM-2，训练更快且不容易过拟合

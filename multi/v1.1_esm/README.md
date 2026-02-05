# ARG 多分类模型 v1.1 - ESM-2 + BiLSTM

## 改进点

相比 v1.0 版本，v1.1 引入了以下改进：

1. **ESM-2 预训练模型**：使用 Facebook 的 ESM-2 作为特征提取器
   - 蛋白质语言模型预训练，捕获进化信息
   - 支持多种模型尺寸

2. **改进的类别不平衡处理**：
   - Focal Loss with Label Smoothing
   - 加权随机采样 (WeightedRandomSampler)
   - 平方根类别权重

3. **注意力机制**：多头注意力池化，聚焦关键序列区域

4. **维度缩减层**：将 ESM-2 的高维输出降至适当维度，再输入 BiLSTM

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

### 配置说明

编辑 `train.py` 中的 `Config` 类：

```python
class Config:
    # ESM-2 模型选择
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    ESM_FREEZE_LAYERS = 4  # 冻结前4层
    
    # 训练参数
    MAX_LENGTH = 1000
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 100
    PATIENCE = 20
    
    # Focal Loss 参数
    FOCAL_GAMMA = 1.0
    LABEL_SMOOTHING = 0.1
    
    # 类别处理
    MIN_SAMPLES = 30  # 少于30样本的类别合并为"Others"
    
    # 数据路径（必须修改）
    FASTA_FILE = "/path/to/your/arg_sequences.fasta"
```

### FASTA 格式要求

序列的 header 需要包含类别标签，格式如下：

```
>protein_id|source|gene_name|drug_class|...
MAKQIS...
```

其中 `drug_class` 位置应为抗性类别（如 beta-lactam, aminoglycoside 等）。

## 使用模型进行分类

```bash
cd model_test

# 单个文件
python classify.py \
    --model ../well-trained/esm2_bilstm_multi_YYYYMMDD_HHMM.pth \
    --input /path/to/input.fasta \
    --output ./results

# 批量处理目录
python classify.py \
    --model ../well-trained/esm2_bilstm_multi_YYYYMMDD_HHMM.pth \
    --input /path/to/input_directory/ \
    --output ./results \
    --confidence_threshold 0.5
```

### 分类参数

- `--model`: 训练好的模型路径
- `--input`: 输入 FASTA 文件或目录
- `--output`: 输出目录
- `--confidence_threshold`: 置信度阈值，过滤低置信度预测
- `--device`: 使用 cuda 或 cpu
- `--batch_size`: 批大小（可选）

## 输出结果

1. `classification_results.csv` - 主要预测结果
2. `classification_results_detailed.json` - 包含 Top-3 预测的详细信息

CSV 格式：
| 列名 | 说明 |
|------|------|
| id | 序列 ID |
| source_file | 源文件 |
| predicted_class | 预测类别 |
| confidence | 置信度 |
| sequence | 序列（截断） |

JSON 格式包含额外的 `top3_predictions` 字段。

## 模型架构

```
Input Sequence
    ↓
ESM-2 Tokenizer
    ↓
ESM-2 Model
    ↓ (batch, seq_len, esm_hidden)
Dimension Reduction (Linear 768->256)
    ↓
BiLSTM (2 layers, bidirectional)
    ↓ (batch, seq_len, hidden*2)
Attention Pooling
    ↓ (batch, hidden*2)
Classifier (Linear -> ReLU -> Dropout -> Linear)
    ↓
Class Probabilities (Softmax)
```

## 类别处理

### 稀有类别合并

默认将样本数少于 `MIN_SAMPLES`（默认30）的类别合并为 "Others"。

### 类别权重

使用平方根倒数权重：
```
weight[i] = sqrt(total_samples / (num_classes * count[i]))
```

权重范围限制在 [0.5, 5.0] 之间。

## 性能指标

| 指标 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Macro F1 | 各类别 F1 的平均（不考虑样本数） |
| Weighted F1 | 加权平均 F1 |
| Per-class F1 | 每个类别的 F1 分数 |

## 注意事项

1. **类别不平衡**：多分类任务中不同类别样本数差异大，使用 Focal Loss 和加权采样缓解
2. **长序列**：MAX_LENGTH 默认为 1000，超长序列会被截断
3. **未知类别**：如果输入序列来自训练时未见的类别，模型可能会预测错误

## 与 v1.0 对比

| 特性 | v1.0 | v1.1 |
|------|------|------|
| 特征提取 | One-hot | ESM-2 预训练 |
| 类别采样 | 简单 shuffle | 加权随机采样 |
| 损失函数 | Focal Loss | Focal + Label Smoothing |
| 池化方式 | Max + Avg | Attention |
| 预计 Macro F1 | ~75% | ~85-90% |

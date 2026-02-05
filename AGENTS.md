# ARG 深度学习预测项目 - AGENTS.md

> 本项目旨在基于深度学习的方法，对全长的氨基酸序列进行抗性基因（ARG）识别（二分类）和分类（多分类）

---

## 📌 项目背景

### 任务目标
- **二分类任务**：识别输入序列是否为抗性基因（ARG）
- **多分类任务**：预测抗性基因序列属于哪个抗性基因类别

### 数据集概况
- **总序列数**：17,345 条非冗余序列
- **数据来源**：CARD, AMRFinder, SARG, MegaRes, ResFinder 五大数据库
- **类别分布**：30 个抗性基因类别
  - Beta-lactam resistance: 7,336 (42.3%)
  - Multidrug resistance: 3,548 (20.5%)
  - MLS resistance: 1,814 (10.5%)
  - ... (其他27个类别)

---

## 📁 项目结构

```
/home/lanty/Documents/study/model_k/
├── binary/                          # ARG 识别二分类模型
│   ├── model_train/
│   │   └── train.ipynb              # BiLSTM 二分类训练代码
│   └── model_test/
│       └── predict.ipynb            # 批量预测脚本
├── multi/                           # ARG 类别多分类模型
│   ├── model_train/
│   │   └── train.ipynb              # BiLSTM 多分类训练代码
│   └── model_test/
│       └── classify.ipynb           # 批量分类脚本
├── data/                            # 数据集
│   ├── ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta
│   └── database_construct_description.md
└── AGENTS.md                        # 本文件
```

---

## 🏗️ 当前模型架构

### 二分类模型 (BiLSTM-Binary)

```python
MODEL_CONFIG = {
    'vocab_size': 22,        # 20氨基酸 + X + PAD
    'embedding_dim': 48,     # Embedding维度
    'hidden_size': 48,       # LSTM隐藏层维度
    'num_layers': 1,         # LSTM层数
    'dropout': 0.5,
    'max_length': 1000,
}

TRAIN_CONFIG = {
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 5e-3,
    'epochs': 100,
    'patience': 15,
    'pos_neg_ratio': 3,      # 阴性样本倍数
}
```

**架构流程**：
```
Input (seq_len) 
  -> Embedding (vocab=22, dim=48)
  -> BiLSTM (1层, hidden=48)
  -> Global Pooling (Max + Avg)
  -> Dropout(0.5)
  -> Linear(192 -> 48) -> ReLU
  -> Dropout(0.5)
  -> Linear(48 -> 1) -> Sigmoid
```

**技术亮点**：
- 使用 `pos_weight` 处理类别不平衡
- `ReduceLROnPlateau` 学习率调度
- 早停机制 (patience=15)
- 保存完整模型配置到 checkpoint

### 多分类模型 (BiLSTM-Multi)

```python
MODEL_CONFIG = {
    'embedding_size': 21,    # One-hot编码 (20氨基酸 + PAD)
    'hidden_size': 128,      # LSTM隐藏层
    'num_layers': 2,         # LSTM层数
    'dropout': 0.4,
}

TRAIN_CONFIG = {
    'batch_size': 256,
    'lr': 0.002,
    'warmup_epochs': 5,
    'epochs': 150,
    'patience': 25,
    'min_samples': 50,       # 稀有类别合并阈值
    'focal_gamma': 0.5,      # Focal Loss gamma
    'label_smoothing': 0.1,
}
```

**架构流程**：
```
Input (seq_len)
  -> One-hot Encoding (21维)
  -> BiLSTM (2层, hidden=128)
  -> Global Pooling (Max + Avg)
  -> Dropout(0.4)
  -> Linear(512 -> 128) -> ReLU
  -> Dropout(0.4)
  -> Linear(128 -> num_classes) -> Softmax
```

**技术亮点**：
- Focal Loss 处理类别不平衡
- 标签平滑 (Label Smoothing)
- Warmup + Cosine Annealing 学习率调度
- 稀有类别自动合并机制

---

## 🎯 优化路线图

### 迭代目标
目标：超越现有 SOTA 工具 (DeepARG, RGI, sARG2.0 等)，达到 **识别 >95%, 分类 >90%**

### SOTA 对比基准

| 工具 | 方法 | 识别准确率 | 分类准确率 |
|------|------|-----------|-----------|
| DeepARG | CNN + Word2Vec | ~85% | ~75% |
| ARG-ANNOT | 基于BLAST | ~90% | ~80% |
| RGI (CARD) | 基于规则 | ~95% | ~85% |
| sARG2.0 | 机器学习 | ~88% | ~78% |

---

## 🚀 优化方向

### 🔴 高优先级 (v1.1 - v1.3)

#### v1.1: 引入预训练蛋白质语言模型
- **问题**：当前简单 Embedding/One-hot 表达能力有限
- **方案**：集成 ESM-2 或 ProtTrans 预训练模型
- **预期收益**：+5-15% 准确率
- **具体实施**：
  1. 使用 ESM-2 提取序列特征
  2. 冻结/微调预训练权重
  3. 与现有 BiLSTM 融合

#### v1.2: 增强特征工程
- **问题**：仅使用氨基酸序列信息
- **方案**：添加物理化学性质编码
  - AAC (Amino Acid Composition)
  - DPC (Dipeptide Composition)
  - CTD (Composition, Transition, Distribution)
- **预期收益**：+2-5% 准确率

#### v1.3: 升级模型架构
- **问题**：BiLSTM 过于简单，缺乏全局注意力
- **方案**：
  - 引入 Transformer Encoder
  - 或 CNN + BiLSTM 混合架构
  - 添加 Self-Attention / SE Block
- **预期收益**：+3-8% 准确率

### 🟡 中等优先级 (v1.4 - v1.5)

#### v1.4: 数据增强与正则化
- 随机 Mask 氨基酸
- 序列随机截断/填充
- Mixup/CutMix 等样本混合技术

#### v1.5: 多模型集成
- 训练多个不同初始化/结构的模型
- 使用投票或堆叠法集成

### 🟢 长期优化 (v2.0+)

#### v2.0: 多任务学习
- 联合训练识别+分类任务
- 共享底层表示，任务-specific 顶层

#### v2.1: 处理超长序列
- 当前 max_length 固定为 1000/95分位数
- 引入分层注意力或分段处理机制

#### v2.2: 模型轻量化
- 知识蒸馏
- 量化/剪枝
- 支持边缘部署

---

## 📊 实验记录模板

每次迭代请记录以下信息：

```markdown
### 版本: vX.Y
**日期**: YYYY-MM-DD

#### 改动内容
- 修改1
- 修改2

#### 实验配置
- 学习率:
- Batch size:
- 其他超参:

#### 实验结果
| 指标 | 训练集 | 验证集 | 测试集 |
|------|-------|-------|-------|
| Accuracy |  |  |  |
| Precision |  |  |  |
| Recall |  |  |  |
| F1 Score |  |  |  |
| AUC-ROC |  |  |  |

#### 结论与下一步
- 
```

---

## 🔧 开发规范

### 代码组织
1. 训练代码统一放在 `model_train/` 目录
2. 测试/推理代码统一放在 `model_test/` 目录
3. 每个版本创建独立子目录，如 `binary/v1.1_esm/`

### 模型保存规范
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': MODEL_CONFIG,           # 必须包含
    'train_config': TRAIN_CONFIG,     # 可选
    'class_names': class_names,       # 多分类必须
    'label_to_idx': label_to_idx,     # 多分类必须
    'max_length': max_length,         # 必须包含
    'performance': {                  # 记录性能
        'val_acc': best_acc,
        'val_f1': best_f1,
    }
}
```

### 命名规范
- 模型文件：`{model_type}_{version}_{date}.pth`
  - 示例：`bilstm_v1.1_20250205.pth`
- 图表目录：`figures/v{X.Y}/`
- 日志文件：`logs/train_v{X.Y}_{date}.log`

---

## 📚 参考资料

### 相关论文
1. DeepARG: a deep learning approach for predicting antibiotic resistance genes from metagenomic data
2. ESM-2: Language models of protein sequences at the scale of evolution
3. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning

### 相关工具
- [ESM-2 GitHub](https://github.com/facebookresearch/esm)
- [ProtTrans](https://github.com/agemagician/ProtTrans)
- [CARD](https://card.mcmaster.ca/)

---

## 💬 协作记录

> 记录与项目相关的重要讨论和决策

### 2025-02-05
- **项目初始化完成**
- 确定了迭代优化的整体方向
- 下一步：开始 v1.1 版本的 ESM-2 集成

---

*最后更新: 2025-02-05*

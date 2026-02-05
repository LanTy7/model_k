# ARG 深度学习预测项目 - AGENTS.md

> 基于深度学习的方法，对全长氨基酸序列进行抗性基因（ARG）识别（二分类）和分类（多分类）
> 目标：超越 SOTA（DeepARG, RGI 等），达到识别 >95%, 分类 >90%

---

## 📋 项目规范（必读）

### 记录体系：双层日志

| 文件 | 用途 | 更新时机 |
|------|------|---------|
| `AGENTS.md` | 稳定共识：架构、规则、路线图 | 架构变更时 |
| `WORKLOG.md` | 工作日志：每次代码/实验的具体变更 | 每次迭代后追加 |

**铁律**：任何影响复现的变更（代码、数据、配置、评估协议）必须写入 `WORKLOG.md`

### WORKLOG 记录模板

```markdown
## YYYY-MM-DD HH:MM - 一句话概括本次目的

### 变更
- **文件**: `path/to/file`
  - 要点1：改了什么
  - 要点2：为什么改

### 实验
- **数据**: `/path/to/data` (版本/MD5)
- **配置**: `batch=32`, `lr=1e-4`, `max_length=1000`
- **结果**: Overall Acc: 0.92 | Macro-F1: 0.88 | Others F1: 0.65

### 结论/下一步
- 结论：xxx
- 下一步：xxx
```

### 命名规范
- 模型：`{type}_{version}_{date}.pth` 例：`esm2_bilstm_v1.1_20260205.pth`
- 版本目录：`binary/v1.1_esm/`, `multi/v1.1_esm/`

---

## 📁 项目结构

```
model_k/
├── AGENTS.md              # 本文档：规则与架构
├── WORKLOG.md             # 工作日志：实验记录
├── data/
│   └── *.fasta            # 数据集（17,345条，30类）
├── binary/                # 二分类：是否ARG
│   ├── v1.0/              # BiLSTM 基线（ipynb）
│   └── v1.1_esm/          # ESM-2 + BiLSTM
│       ├── model_train/train.py
│       ├── model_test/predict.py
│       └── README.md
└── multi/                 # 多分类：ARG类别（30类）
    ├── v1.0/              # BiLSTM 基线
    └── v1.1_esm/          # ESM-2 + BiLSTM
        ├── model_train/train.py
        ├── model_test/classify.py
        └── README.md
```

---

## 🏗️ 模型版本

### v1.0（基线）- BiLSTM
| 任务 | 架构 | 特征 | 预期指标 |
|------|------|------|---------|
| 二分类 | Embedding(48) → BiLSTM → Pooling | 简单Embedding | Acc ~85% |
| 多分类 | One-hot(21) → BiLSTM → Pooling | One-hot编码 | Macro-F1 ~75% |

### v1.1（当前）- ESM-2 + BiLSTM  
**改进**：引入预训练蛋白质语言模型 ESM-2

```
Input → ESM-2 Tokenizer → ESM-2 (freeze N layers) → Dim Reduction 
→ BiLSTM → Attention Pooling → Classifier
```

| 组件 | 配置 |
|------|------|
| ESM-2 | `esm2_t6_8M_UR50D` (小) / `t12_35M` (中) / `t30_150M` (大) |
| Freeze | 前4-6层冻结，其余微调 |
| BiLSTM | hidden=128, layers=2, dropout=0.3 |
| 优化器 | 分层LR：ESM-2 1e-5，其他 1e-4 |
| 损失 | BCE/Focal Loss + Label Smoothing |

**预期提升**：+5-15% 准确率

---

## 🎯 优化路线图

| 版本 | 目标 | 关键技术 | 预期收益 |
|------|------|---------|---------|
| v1.1 | ESM-2集成 | 预训练蛋白质语言模型 | +5-15% |
| v1.2 | 特征增强 | AAC/DPC/CTD物理化学特征 | +2-5% |
| v1.3 | 架构升级 | Transformer / CNN+BiLSTM | +3-8% |
| v1.4 | 数据增强 | Mask/截断/Mixup | +2-4% |
| v1.5 | 模型集成 | 多模型投票 | +2-3% |
| v2.0 | 多任务 | 联合训练识别+分类 | 整体优化 |

---

## 📊 SOTA 对比基准

| 工具 | 方法 | 识别 | 分类 |
|------|------|------|------|
| DeepARG | CNN+Word2Vec | ~85% | ~75% |
| RGI (CARD) | 规则 | ~95% | ~85% |
| **我们的目标** | **ESM-2+BiLSTM** | **>95%** | **>90%** |

---

## 💬 关键决策记录

### 2026-02-05
- 确定双层记录体系（AGENTS + WORKLOG）
- v1.1 代码开发完成，待训练验证
- **待解决**：二分类需要准备阴性样本

---

## 📚 参考

- ESM-2: [facebookresearch/esm](https://github.com/facebookresearch/esm)
- 数据集：CARD, AMRFinder, SARG, MegaRes, ResFinder
- 总序列：17,345条，30个抗性类别

---

*简明版 AGENTS.md | 详细内容见各版本 README.md 和 WORKLOG.md*

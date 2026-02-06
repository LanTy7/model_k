# 模型真实场景测试与验证协议（论文可复现版本）

> 定义在真实数据上测试模型的标准流程，确保结果可复现、可量化、可写入论文方法学部分。

---

## 1. 测试目标与输入

### 1.1 目标
- **二分类**：ORF 是否为 ARG
- **多分类**：ARG 属于哪个抗性类别

### 1.2 输入数据
- `.faa` 文件（来自 `contig → Prodigal → ORF proteins`）
- 参考 ARG 数据库：使用与训练标签体系一致的库（如 `data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta`）

### 1.3 测试集规模建议
| 场景 | 规模 | 用途 |
|------|------|------|
| 快速验证 | 1-5万条 | 开发调试 |
| 主实验 | 10-50万条 | 论文主要结果 |
| 大规模验证 | 100万条 | 可扩展性证明（可选） |

**注意**：若原始数据达亿级（如 3.43亿条），必须分层抽样，不可全量测试。

---

## 2. 两类评估场景（论文必须区分）

### 2.1 应用示例（Case Study）
- **流程**：`.faa` → 二分类 → 多分类 → 仅对预测 ARG 做同源验证
- **用途**：展示模型输出样例、发现候选 ARG
- **局限**：无法统计假阴性（漏检），不能反映真实召回率

### 2.2 真实性能评估（主结果）
- **核心**：先给**全部 ORF**构建参考标签，再与模型输出对比
- **产出**：完整的混淆矩阵（TP/FP/TN/FN）、Macro-F1、每类 F1

---

## 3. 参考标签（Silver Standard）构建

### 3.1 比对工具选择（论文需明确）
| 工具 | 特点 | 推荐场景 |
|------|------|---------|
| **DIAMOND blastp** | 速度快 | 大规模 ORF（>10万条）⭐推荐 |
| **MMseqs2** | 速度极快，敏感度高 | 超大规模（>100万条） |
| **BLASTP** | 经典，准确度基准 | 小规模验证（<1万条） |

### 3.2 高置信命中阈值

```
e-value ≤ 1e-5
pident ≥ 80%（或 70%，严格度需在论文说明）
覆盖度（满足其一）：
  - qcov ≥ 0.8（alignment / query）
  - scov ≥ 0.8（alignment / subject）
```

**说明**：
- 主结果使用**严格阈值**（高置信）
- 补充材料可增加**宽松阈值**（pident≥60%）的敏感性分析

### 3.3 标签赋值规则

| 情况 | 二分类标签 | 多分类标签 | 处理方式 |
|------|-----------|-----------|---------|
| 无高置信命中 | `non-ARG` | - | 可作为阴性样本 |
| 有单一高置信命中 | `ARG` | 取 best-hit 类别 | 正常参与评估 |
| 多命中但类别一致 | `ARG` | 该共同类别 | 正常参与评估 |
| 多命中但类别冲突 | `ARG` | `ambiguous` | 从多分类评估中剔除 |
| 无法确定（远缘/碎片） | `unlabeled` | - | 单独分析，勿硬当阴性 |

---

## 4. 防止训练-测试同源泄漏（必须）

### 4.1 同源去重方案
```bash
# 步骤：测试 ORF 与训练集比对
diamond blastp --query test_orfs.faa \
               --db train_arg_db.dmnd \
               --outfmt 6 qseqid sseqid pident length qlen slen \
               --out homology_check.tsv

# 剔除阈值：pident ≥ 90% 且 (qcov ≥ 90% 或 scov ≥ 90%)
# 被剔除的标记为 `seen-like`，剩余的为 `novel-like`
```

### 4.2 按来源 Holdout（更严格）
- 按数据库来源（sarg/ncbi/megares/card/resf）拆分
- 测试仅使用 holdout 来源的数据

### 4.3 报告要求
- 主结果在 `novel-like` 子集上报告（反映泛化能力）
- 可补充 `seen-like` vs `novel-like` 对比表

---

## 5. 阈值选择与校准

### 5.1 二分类阈值策略

**方案A：验证集最优阈值**
```python
# 在验证集上确定阈值，固定后应用于测试集
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    preds = (val_probs >= thresh).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# 论文报告："The classification threshold (0.xx) was determined 
# on the validation set to maximize F1-score"
```

**方案B：固定阈值（0.5）**
- 优点：简单，可复现
- 缺点：可能非最优

**推荐**：方案A，并在补充材料报告阈值敏感性曲线

### 5.2 多分类置信度阈值（可选）
```python
# 对低置信度预测标记为 "uncertain"
if max_prob < 0.6:
    prediction = "uncertain"  # 从评估中剔除或单独统计
```

---

## 6. 评估指标（论文必须报告）

### 6.1 二分类指标
| 指标 | 公式 | 意义 |
|------|------|------|
| Precision | TP/(TP+FP) | 预测为ARG的中多少是真的 |
| Recall (Sensitivity) | TP/(TP+FN) | 真实ARG中被找回的比例 |
| Specificity | TN/(TN+FP) | 非ARG中被正确排除的比例 |
| F1 | 2·P·R/(P+R) | 精确度和召回率的平衡 |
| PR-AUC | - | 不同阈值下的综合性能 |
| ROC-AUC | - | 类别分离能力 |

### 6.2 多分类指标
| 指标 | 说明 |
|------|------|
| **Macro-F1** | 各类F1的平均（核心指标）⭐ |
| Weighted-F1 | 按类别样本数加权的F1 |
| Per-class F1 | 每类的Precision/Recall/F1 |
| 混淆矩阵 | 可视化类别混淆情况 |

**特别注意**：
- 对 `Others`/长尾类单独讨论
- 避免"垃圾桶类"吞噬长尾后指标虚高

### 6.3 端到端（Binary → Multi）错误分解

```
端到端正确：二分类判为ARG 且 多分类类别正确

端到端错误类型：
├─ 漏检（Miss）：参考为ARG，但二分类预测 non-ARG
├─ 误检（False Alarm）：参考为 non-ARG，但二分类预测 ARG
└─ 定类错（Misclass）：二分类对，但多分类类别错
```

**论文呈现**：错误分解饼图或堆叠柱状图

---

## 7. 与 SOTA 工具对比（必须）

### 7.1 对比工具
| 工具 | 类型 | 说明 |
|------|------|------|
| **RGI** | 基于规则 | 目前最权威的ARG注释工具 |
| **DeepARG** | 深度学习 | 基于CNN+Word2Vec的SOTA |
| **ARG-ANNOT** | 基于同源 | 经典工具 |

### 7.2 对比维度
- 敏感性（Sensitivity）：谁能找到更多ARG？
- 特异性（Specificity）：谁更少误报？
- 运行速度：处理10万条序列所需时间
- 发现novel ARGs的能力（在novel-like子集上对比）

### 7.3 统计显著性
- 使用 McNemar's test 比较两个模型在相同样本上的差异显著性
- 报告 p-value（p < 0.05 认为差异显著）

---

## 8. 结果呈现建议

### 8.1 主表（必需）
| 评估子集 | Binary F1 | Binary Recall | Macro-F1 | Weighted-F1 |
|---------|-----------|---------------|----------|-------------|
| All | 0.xx | 0.xx | 0.xx | 0.xx |
| novel-like | 0.xx | 0.xx | 0.xx | 0.xx |
| seen-like | 0.xx | 0.xx | 0.xx | 0.xx |

### 8.2 子表（推荐）
- **按ORF长度分段**：`<200aa`、`200-400aa`、`>400aa`
- **按类别分组**：各类别的 F1 对比

### 8.3 图示
- **PR曲线**：展示不同阈值下的 Precision-Recall 权衡
- **混淆矩阵**：多分类结果的heatmap
- **错误分解**：端到端错误类型分布

### 8.4 案例展示
列出若干代表性预测：
- 高置信度正确（典型ARG）
- 高置信度但无同源（候选novel ARG）
- 低置信度（模型不确定）

---

## 9. 实施 checklist

- [ ] 测试集与训练集进行同源去重（pident<90%）
- [ ] 明确参考标签构建的阈值（e-value, pident, coverage）
- [ ] 确定二分类阈值选择策略（验证集最优 or 固定0.5）
- [ ] 运行至少1个SOTA工具（RGI或DeepARG）进行对比
- [ ] 报告 Macro-F1 和 Per-class F1
- [ ] 对 `unlabeled` 样本单独分析，不硬当阴性
- [ ] 分层抽样（若原始数据>100万条）
- [ ] 所有阈值、参数在论文方法学部分明确说明

---

## 10. 常见问题（FAQ）

**Q1: 测试集需要多少阴性样本？**
> 建议阳性:阴性 = 1:1 到 1:3。若真实数据阴性比例未知，可分层抽样后混入已知非ARG作为对照。

**Q2: ORF不完整（截断）怎么办？**
> 按长度分层报告。短ORF（<100aa）预测置信度通常较低，可在论文中讨论。

**Q3: 模型预测为ARG但无同源，是假阳性还是novel ARG？**
> 标记为"候选novel ARG"，结合以下证据判断：
> - 序列是否包含已知ARG功能域（如使用InterProScan）
> - 所在contig是否有其他已知ARG（共定位证据）
> - 表达数据（如可用）是否支持功能活性

---

*文档版本: v1.0 | 最后更新: 2026-02-05*

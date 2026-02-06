# 工作日志 (WORKLOG)

> 记录每一次代码改动与实验结果，保持可追溯、可复现

---

## 2026-02-05 01:27 - 项目初始化 + v1.1 代码开发（ESM-2 + BiLSTM）

### 变更

- **文件**: `/home/lanty/Documents/study/model_k/AGENTS.md`
  - 创建项目文档，记录背景、架构、优化路线图
  - 新增"日志与记录（规则）"章节，确立双层记录体系
- **文件**: `/home/lanty/Documents/study/model_k/binary/v1.1_esm/model_train/train.py`
  - 新建：ESM-2 + BiLSTM 二分类训练脚本
  - 特性：支持分层学习率、Attention Pooling、早停机制
- **文件**: `/home/lanty/Documents/study/model_k/binary/v1.1_esm/model_test/predict.py`
  - 新建：批量预测脚本，支持单文件/目录批量处理
  - 输出：CSV 报告 + 预测的 FASTA 文件
- **文件**: `/home/lanty/Documents/study/model_k/multi/v1.1_esm/model_train/train.py`
  - 新建：ESM-2 + BiLSTM 多分类训练脚本
  - 特性：Focal Loss + Label Smoothing + WeightedRandomSampler
- **文件**: `/home/lanty/Documents/study/model_k/multi/v1.1_esm/model_test/classify.py`
  - 新建：批量分类脚本，支持 Top-3 预测输出
  - 输出：CSV 主报告 + JSON 详细报告
- **文件**: `binary/v1.1_esm/requirements.txt`, `multi/v1.1_esm/requirements.txt`
  - 新建：依赖列表（torch, transformers, biopython, scikit-learn 等）
- **文件**: `binary/v1.1_esm/README.md`, `multi/v1.1_esm/README.md`
  - 新建：各版本的使用说明和架构说明
- **文件**: `binary/v1.1_esm/run_train.sh`, `multi/v1.1_esm/run_train.sh`
  - 新建：一键训练启动脚本

### 实验

- **数据**: 
  - ARG: `/home/lanty/Documents/study/model_k/data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta` (17,345条)
  - 非ARG: 待准备（当前代码中需要配置 `NON_ARG_FILE`）
- **配置**:
  - `ESM_MODEL_NAME=facebook/esm2_t6_8M_UR50D`
  - `ESM_FREEZE_LAYERS=4`
  - `MAX_LENGTH=1000`
  - `BATCH_SIZE=32`
  - `LR=1e-4` (ESM-2: 1e-5)
  - `EPOCHS=100`, `PATIENCE=15`
- **结果**:
  - 尚未训练，代码已就绪

### 结论/下一步

- **结论**: v1.1 版本代码开发完成，具备 ESM-2 集成、Attention Pooling、分层学习率等特性
- **下一步**: 
  准备二分类的阴性样本数据（非ARG序列）
- 运行多分类模型训练（数据已就绪）
- 记录首次训练实验结果

---

## 2026-02-06 22:56 - Bug修复：训练脚本日志路径错误

### 变更

- **文件**: `/home/lanty/Documents/study/model_k/multi/v1.1_esm/model_train/train.py`
  - 修复：日志保存路径从 `logs/` 改为 `../logs/`
  - 添加：`os.makedirs('../logs', exist_ok=True)` 自动创建目录
  - 原因：代码在 `model_train/` 目录运行，原相对路径导致 `FileNotFoundError`

- **文件**: `/home/lanty/Documents/study/model_k/binary/v1.1_esm/model_train/train.py`
  - 同上修复

- **文件**: `/home/lanty/Documents/study/model_k/multi/v1.1_esm/run.sh`, `binary/v1.1_esm/run.sh`
  - 创建：集群直接运行脚本（无需 SLURM）
  - 功能：自动激活 conda 环境、后台运行、输出 PID 和日志路径

- **文件**: `/home/lanty/Documents/study/model_k/multi/v1.1_esm/submit_job.sh`, `binary/v1.1_esm/submit_job.sh`
  - 创建：SLURM 作业脚本（备用）

- **文件**: `/home/lanty/Documents/study/model_k/environment.yml`, `multi/v1.1_esm/environment.yml`, `binary/v1.1_esm/environment.yml`
  - 创建：Conda 环境配置文件，支持 CUDA 12.1

### 实验

- **数据**: 同上
- **配置**: 
  - 改为 `facebook/esm2_t12_35M_UR50D`（35M参数，平衡性能与速度）
  - GPU: Tesla V100S 32GB，可用显存 ~24GB
- **结果**: 
  - 首次启动训练，因日志路径错误失败
  - 修复后重新启动，训练进行中

### 结论/下一步

- **结论**: 日志路径 Bug 已修复，多分类模型训练已启动
- **下一步**: 
  - 监控训练日志，确认正常收敛
  - 等待训练完成（预计 4-6 小时）
  - 准备二分类阴性样本

---

*记录格式遵循 AGENTS.md 中的"日志与记录（规则）"章节*
*注：修正日期错误，正确年份为 2026*
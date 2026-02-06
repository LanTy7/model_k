#!/usr/bin/env python3
"""
ARG 多分类模型 v1.1 - ESM-2 + BiLSTM
使用预训练蛋白质语言模型 ESM-2 提取特征，对抗性基因进行30分类

改进点：
1. 引入 ESM-2 预训练模型作为特征提取器
2. Focal Loss + Label Smoothing 处理类别不平衡
3. 类别重采样策略
4. 支持合并稀有类别
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from collections import Counter
from datetime import datetime
from tqdm import tqdm

# 设置日志
os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'../logs/train_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 配置参数 ====================

class Config:
    """模型和训练配置"""
    # ESM-2 模型配置
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    ESM_FREEZE_LAYERS = 4  # 冻结前 N 层
    MAX_LENGTH = 1000      # 最大序列长度
    
    # BiLSTM 配置
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # Attention 配置
    ATTENTION_HEADS = 8
    
    # 训练配置
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    PATIENCE = 20          # 多分类任务更复杂，增加耐心值
    
    # Focal Loss 配置
    FOCAL_GAMMA = 1.0
    LABEL_SMOOTHING = 0.1
    
    # 类别处理
    MIN_SAMPLES = 30       # 少于这个数量的类别合并为 "Others"
    
    # 数据路径
    FASTA_FILE = "/home/lanty/Documents/study/model_k/data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta"
    SAVE_DIR = "./well-trained"
    FIG_DIR = "./figures"
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 随机种子
    SEED = 42


def set_seed(seed=42):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 数据预处理 ====================

class ESMTokenizer:
    """ESM-2 模型专用的 tokenizer"""
    
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.special_tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>']
        
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        for i, token in enumerate(self.special_tokens):
            self.token_to_idx[token] = i
            self.idx_to_token[i] = token
        
        for i, aa in enumerate(self.amino_acids):
            idx = i + 5
            self.token_to_idx[aa] = idx
            self.idx_to_token[idx] = aa
        
        self.vocab_size = len(self.token_to_idx)
        self.pad_token_id = self.token_to_idx['<pad>']
        self.cls_token_id = self.token_to_idx['<cls>']
        self.eos_token_id = self.token_to_idx['<eos>']
        self.unk_token_id = self.token_to_idx['<unk>']
    
    def encode(self, sequence, max_length=None):
        """将氨基酸序列编码为 token indices"""
        sequence = ''.join([aa for aa in sequence.upper() if aa in self.amino_acids])
        
        if max_length:
            sequence = sequence[:max_length-2]
        
        tokens = [self.cls_token_id]
        tokens += [self.token_to_idx.get(aa, self.unk_token_id) for aa in sequence]
        tokens.append(self.eos_token_id)
        
        if max_length:
            if len(tokens) < max_length:
                tokens += [self.pad_token_id] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
        
        return tokens


class ARGDataset(Dataset):
    """ARG 多分类数据集"""
    
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self.tokenizer.encode(seq, self.max_length)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': min(len(seq) + 2, self.max_length)
        }


def parse_fasta_header(header):
    """
    解析 FASTA header，提取类别标签
    假设格式: >id|...|label|...
    根据实际数据格式调整
    """
    parts = header.split('|')
    
    # 尝试不同位置查找标签
    # 格式示例: >protein_id|source|gene_name|drug_class|...
    if len(parts) >= 4:
        label = parts[3].strip()
        if label and label.lower() not in ['unknown', 'na', '']:
            return label
    
    # 如果找不到，尝试其他位置
    for part in parts:
        part = part.strip()
        if 'resistance' in part.lower() or part.lower() in [
            'beta-lactam', 'aminoglycoside', 'tetracycline', 'macrolide',
            'fluoroquinolone', 'phenicol', 'sulfonamide', 'trimethoprim',
            'multidrug', 'glycopeptide', ' MLS'
        ]:
            return part
    
    return None


def load_and_preprocess_data(config):
    """
    加载并预处理数据
    返回: sequences, labels, class_names, label_to_idx, idx_to_label
    """
    logger.info(f"Loading data from {config.FASTA_FILE}")
    
    sequences = []
    raw_labels = []
    
    for record in SeqIO.parse(config.FASTA_FILE, "fasta"):
        seq = str(record.seq).upper()
        
        # 跳过太短的序列
        if len(seq) < 10:
            continue
        
        # 解析标签
        label = parse_fasta_header(record.description)
        
        if label is None:
            # 尝试从 description 的其他部分解析
            desc_lower = record.description.lower()
            if 'beta-lactam' in desc_lower:
                label = 'beta-lactam'
            elif 'multidrug' in desc_lower or 'multi-drug' in desc_lower:
                label = 'multidrug'
            elif 'aminoglycoside' in desc_lower:
                label = 'aminoglycoside'
            elif 'tetracycline' in desc_lower:
                label = 'tetracycline'
            elif 'macrolide' in desc_lower:
                label = 'macrolide'
            elif 'mls' in desc_lower:
                label = 'MLS'
            else:
                # 无法识别的标签，跳过或使用默认标签
                continue
        
        sequences.append(seq)
        raw_labels.append(label)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # 统计类别分布
    label_counts = Counter(raw_labels)
    logger.info(f"Found {len(label_counts)} unique labels")
    logger.info("Top 10 classes:")
    for label, count in label_counts.most_common(10):
        logger.info(f"  {label}: {count}")
    
    # 合并稀有类别
    rare_classes = [label for label, count in label_counts.items() 
                   if count < config.MIN_SAMPLES]
    
    if rare_classes:
        logger.info(f"Merging {len(rare_classes)} rare classes into 'Others'")
        labels = ['Others' if label in rare_classes else label for label in raw_labels]
    else:
        labels = raw_labels
    
    # 创建类别映射
    unique_labels = sorted(set(labels))
    if 'Others' in unique_labels:
        unique_labels.remove('Others')
        unique_labels.append('Others')  # Others 放在最后
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # 转换为数字标签
    numeric_labels = [label_to_idx[label] for label in labels]
    
    # 打印最终类别分布
    class_counts = Counter(numeric_labels)
    logger.info(f"\nFinal class distribution ({len(unique_labels)} classes):")
    for idx, label in enumerate(unique_labels):
        logger.info(f"  {idx}: {label} - {class_counts[idx]} samples")
    
    return sequences, numeric_labels, unique_labels, label_to_idx, idx_to_label


# ==================== 模型定义 ====================

class ESM2FeatureExtractor(nn.Module):
    """ESM-2 特征提取器"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", freeze_layers=4):
        super().__init__()
        
        try:
            from transformers import EsmModel
            self.esm = EsmModel.from_pretrained(model_name)
            self.use_hf = True
            logger.info(f"Loaded ESM-2 model: {model_name}")
        except ImportError:
            logger.error("Please install transformers: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load ESM-2: {e}")
            raise
        
        self.hidden_size = self.esm.config.hidden_size
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
            logger.info(f"Frozen first {freeze_layers} layers of ESM-2")
    
    def _freeze_layers(self, n_layers):
        """冻结前 n 层"""
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing"""
    
    def __init__(self, num_classes, gamma=1.0, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        # Label smoothing
        smoothed_targets = torch.zeros_like(inputs)
        smoothed_targets.fill_(self.label_smoothing / (self.num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal loss
        focal_weight = (1 - probs) ** self.gamma
        
        # 计算损失
        loss = -focal_weight * smoothed_targets * log_probs
        
        # 类别权重
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss.sum(dim=1) * weights
        else:
            loss = loss.sum(dim=1)
        
        return loss.mean()


class AttentionPooling(nn.Module):
    """多头注意力池化"""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, mask=None):
        cls_query = x[:, 0:1, :]
        attn_output, attn_weights = self.attention(
            cls_query, x, x,
            key_padding_mask=mask,
            need_weights=True
        )
        return attn_output.squeeze(1), attn_weights


class ESM2BiLSTMClassifier(nn.Module):
    """ESM-2 + BiLSTM + Attention 多分类模型"""
    
    def __init__(self, config, num_classes):
        super().__init__()
        
        self.config = config
        
        # ESM-2 特征提取
        self.esm = ESM2FeatureExtractor(config.ESM_MODEL_NAME, config.ESM_FREEZE_LAYERS)
        
        # 降维层 (ESM-2 输出维度通常很大)
        self.dim_reduction = nn.Linear(self.esm.hidden_size, 256)
        self.dim_activation = nn.ReLU()
        self.dim_dropout = nn.Dropout(0.1)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # Attention
        self.use_attention = True
        if self.use_attention:
            self.attention_pool = AttentionPooling(
                config.LSTM_HIDDEN_SIZE * 2,  # BiLSTM 输出维度
                config.ATTENTION_HEADS
            )
            classifier_input = config.LSTM_HIDDEN_SIZE * 2
        else:
            classifier_input = config.LSTM_HIDDEN_SIZE * 4  # max + avg pooling
        
        # 分类器
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, config.LSTM_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.LSTM_DROPOUT),
            nn.Linear(config.LSTM_HIDDEN_SIZE, num_classes)
        )
    
    def forward(self, input_ids, lengths=None):
        # ESM-2 特征提取
        attention_mask = (input_ids != 1).long()
        esm_output = self.esm(input_ids, attention_mask)
        # esm_output: (batch, seq_len, esm_hidden)
        
        # 降维
        x = self.dim_reduction(esm_output)
        x = self.dim_activation(x)
        x = self.dim_dropout(x)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # 池化
        if self.use_attention:
            features, attn_weights = self.attention_pool(lstm_out)
        else:
            max_pool, _ = torch.max(lstm_out, dim=1)
            avg_pool = torch.mean(lstm_out, dim=1)
            features = torch.cat([max_pool, avg_pool], dim=1)
            attn_weights = None
        
        # 分类
        logits = self.classifier(self.dropout(features))
        
        return logits, attn_weights


# ==================== 训练流程 ====================

def collate_fn(batch):
    """自定义 batch 处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'lengths': lengths
    }


def get_class_weights(labels, method='sqrt'):
    """
    计算类别权重
    method: 'sqrt' - 平方根权重，'inverse' - 倒数权重
    """
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(counts)
    
    weights = np.zeros(num_classes)
    for i in range(num_classes):
        count = counts.get(i, 1)
        if method == 'sqrt':
            weights[i] = np.sqrt(total / (num_classes * count))
        elif method == 'inverse':
            weights[i] = total / (num_classes * count)
        else:
            weights[i] = 1.0
    
    # 限制权重范围
    weights = np.clip(weights, 0.5, 5.0)
    
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels):
    """创建加权采样器以平衡类别"""
    counts = Counter(labels)
    num_samples = len(labels)
    
    # 计算每个样本的权重
    weights = [1.0 / counts[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids, lengths)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, class_names):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            logits, _ = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'preds': all_preds,
        'labels': all_labels
    }


def plot_results(history, eval_results, class_names, save_dir):
    """绘制训练结果"""
    fig = plt.figure(figsize=(16, 12))
    
    # Loss 曲线
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 曲线
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(history['val_macro_f1'], label='Macro F1', color='green')
    ax3.plot(history['val_weighted_f1'], label='Weighted F1', color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 混淆矩阵
    ax4 = plt.subplot(2, 2, 4)
    cm = confusion_matrix(eval_results['labels'], eval_results['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=range(len(class_names)),
                yticklabels=range(len(class_names)))
    ax4.set_title('Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'training_results.pdf'))
    plt.close()
    
    # 每个类别的详细指标图
    fig, ax = plt.subplots(figsize=(14, max(6, len(class_names) * 0.3)))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = eval_results['precision']
    recall = eval_results['recall']
    f1 = eval_results['f1']
    
    ax.barh(x - width, precision, width, label='Precision', alpha=0.8)
    ax.barh(x, recall, width, label='Recall', alpha=0.8)
    ax.barh(x + width, f1, width, label='F1', alpha=0.8)
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Class')
    ax.set_title('Per-Class Metrics')
    ax.set_yticks(x)
    ax.set_yticklabels([f"{i}: {name[:30]}" for i, name in enumerate(class_names)], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()


def main():
    """主训练流程"""
    config = Config()
    set_seed(config.SEED)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Config: {config.__dict__}")
    
    # 加载数据
    sequences, labels, class_names, label_to_idx, idx_to_label = load_and_preprocess_data(config)
    num_classes = len(class_names)
    
    # 划分训练集/验证集
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=config.SEED, stratify=labels
    )
    
    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")
    logger.info(f"Number of classes: {num_classes}")
    
    # 创建 tokenizer 和 dataset
    tokenizer = ESMTokenizer()
    train_dataset = ARGDataset(train_seqs, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = ARGDataset(val_seqs, val_labels, tokenizer, config.MAX_LENGTH)
    
    # 创建加权采样器
    sampler = create_weighted_sampler(train_labels)
    
    # 创建 dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # 使用加权采样
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    # 创建模型
    logger.info("Creating model...")
    model = ESM2BiLSTMClassifier(config, num_classes).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 损失函数
    class_weights = get_class_weights(train_labels, method='sqrt').to(config.DEVICE)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = FocalLoss(
        num_classes=num_classes,
        gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING,
        class_weights=class_weights
    )
    
    # 优化器 - 分层学习率
    esm_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'esm' in name:
                esm_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': esm_params, 'lr': config.LR * 0.1},
        {'params': other_params, 'lr': config.LR}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_macro_f1': [], 'val_weighted_f1': []
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = os.path.join(config.SAVE_DIR, f'esm2_bilstm_multi_{timestamp}.pth')
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(config.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # 验证
        eval_results = evaluate(model, val_loader, criterion, config.DEVICE, class_names)
        val_loss = eval_results['loss']
        val_acc = eval_results['accuracy']
        val_macro_f1 = eval_results['macro_f1']
        val_weighted_f1 = eval_results['weighted_f1']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)
        history['val_weighted_f1'].append(val_weighted_f1)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val Macro F1: {val_macro_f1:.4f}, Weighted F1: {val_weighted_f1:.4f}")
        
        # 学习率调度 - 基于 macro_f1
        scheduler.step(val_macro_f1)
        
        # 早停检查
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'class_names': class_names,
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'performance': eval_results,
            }, save_path)
            logger.info(f"✓ Best model saved (Macro F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"✗ No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                logger.info("Early stopping triggered!")
                break
    
    # 最终评估
    logger.info("\n" + "=" * 60)
    logger.info("Training completed! Loading best model for final evaluation...")
    logger.info("=" * 60)
    
    checkpoint = torch.load(save_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate(model, val_loader, criterion, config.DEVICE, class_names)
    
    logger.info("\nFinal Evaluation Results:")
    logger.info(f"Accuracy: {final_results['accuracy']:.4f}")
    logger.info(f"Macro F1: {final_results['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {final_results['weighted_f1']:.4f}")
    
    # 打印每个类别的详细指标
    logger.info("\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {i}: {class_name[:40]:<40} | "
                   f"Precision: {final_results['precision'][i]:.3f} | "
                   f"Recall: {final_results['recall'][i]:.3f} | "
                   f"F1: {final_results['f1'][i]:.3f} | "
                   f"Support: {final_results['support'][i]:.0f}")
    
    # 打印分类报告
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        final_results['labels'], 
        final_results['preds'],
        target_names=[f"{i}" for i in range(len(class_names))],
        digits=3
    ))
    
    # 绘制结果
    plot_results(history, final_results, class_names, config.FIG_DIR)
    logger.info(f"\nFigures saved to {config.FIG_DIR}")
    
    return save_path


if __name__ == "__main__":
    main()

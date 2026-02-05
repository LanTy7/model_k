#!/usr/bin/env python3
"""
ARG 二分类模型 v1.1 - ESM-2 + BiLSTM
使用预训练蛋白质语言模型 ESM-2 提取特征，结合 BiLSTM 进行分类

改进点：
1. 引入 ESM-2 预训练模型作为特征提取器
2. 支持冻结/微调 ESM-2 权重
3. 添加 Attention 机制增强特征表达
4. 改进的数据增强策略
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
from datetime import datetime
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 配置参数 ====================

class Config:
    """模型和训练配置"""
    # ESM-2 模型配置
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 可选: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D
    ESM_FREEZE_LAYERS = 4  # 冻结前 N 层 ESM-2，0 表示全部微调
    MAX_LENGTH = 1000      # 最大序列长度
    
    # BiLSTM 配置
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # Attention 配置
    ATTENTION_HEADS = 8
    
    # 训练配置
    BATCH_SIZE = 32        # ESM-2 较大，batch_size 适当减小
    LR = 1e-4              # 使用预训练模型，学习率要小
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    PATIENCE = 15
    POS_NEG_RATIO = 3      # 阴性样本倍数
    
    # 数据路径 (需要根据实际路径修改)
    ARG_FILE = "/home/lanty/Documents/study/model_k/data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta"
    NON_ARG_FILE = None    # 如果没有单独的阴性样本文件，将从 ARG 文件中划分
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
        # ESM-2 使用标准的氨基酸字母表
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.special_tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>']
        
        # 构建词汇表
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # 特殊 token
        for i, token in enumerate(self.special_tokens):
            self.token_to_idx[token] = i
            self.idx_to_token[i] = token
        
        # 氨基酸 token (从索引5开始)
        for i, aa in enumerate(self.amino_acids):
            idx = i + 5
            self.token_to_idx[aa] = idx
            self.idx_to_token[idx] = aa
        
        self.vocab_size = len(self.token_to_idx)
        self.pad_token_id = self.token_to_idx['<pad>']
        self.cls_token_id = self.token_to_idx['<cls>']
        self.eos_token_id = self.token_to_idx['<eos>']
        self.unk_token_id = self.token_to_idx['<unk>']
        
        logger.info(f"Tokenizer vocab size: {self.vocab_size}")
    
    def encode(self, sequence, max_length=None):
        """
        将氨基酸序列编码为 token indices
        格式: <cls> + sequence + <eos>
        """
        # 清理序列，只保留标准氨基酸
        sequence = ''.join([aa for aa in sequence.upper() if aa in self.amino_acids])
        
        # 截断
        if max_length:
            sequence = sequence[:max_length-2]  # 预留 <cls> 和 <eos>
        
        # 编码
        tokens = [self.cls_token_id]
        tokens += [self.token_to_idx.get(aa, self.unk_token_id) for aa in sequence]
        tokens.append(self.eos_token_id)
        
        # Padding
        if max_length:
            if len(tokens) < max_length:
                tokens += [self.pad_token_id] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
        
        return tokens
    
    def batch_encode(self, sequences, max_length=None):
        """批量编码"""
        return [self.encode(seq, max_length) for seq in sequences]


class ARGDataset(Dataset):
    """ARG 数据集"""
    
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
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'length': min(len(seq) + 2, self.max_length)  # +2 for <cls> and <eos>
        }


def load_data_from_fasta(fasta_file):
    """从 FASTA 文件加载序列"""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).upper()
        # 过滤掉太短的序列
        if len(seq) >= 10:
            sequences.append(seq)
    return sequences


def prepare_data(config):
    """准备训练数据"""
    logger.info("Loading data...")
    
    # 加载阳性样本 (ARG)
    arg_sequences = load_data_from_fasta(config.ARG_FILE)
    logger.info(f"Loaded {len(arg_sequences)} ARG sequences")
    
    # 如果没有单独的阴性样本文件，从数据集中划分一部分作为阴性
    # 或者需要用户提供阴性样本文件
    if config.NON_ARG_FILE and os.path.exists(config.NON_ARG_FILE):
        non_arg_sequences = load_data_from_fasta(config.NON_ARG_FILE)
        logger.info(f"Loaded {len(non_arg_sequences)} non-ARG sequences")
    else:
        # 暂时从 ARG 中随机采样一些作为阴性样本（仅用于测试）
        # 实际使用时应该提供真正的阴性样本
        logger.warning("No non-ARG file provided! Using random shuffled sequences as negatives (FOR TESTING ONLY)")
        import random
        random.seed(42)
        non_arg_sequences = []
        for seq in arg_sequences[:len(arg_sequences)//5]:
            # 随机打乱序列作为阴性样本
            shuffled = list(seq)
            random.shuffle(shuffled)
            non_arg_sequences.append(''.join(shuffled))
        arg_sequences = arg_sequences[len(arg_sequences)//5:]
    
    # 限制样本数量比例
    n_pos = len(arg_sequences)
    n_neg = min(len(non_arg_sequences), n_pos * config.POS_NEG_RATIO)
    
    if len(non_arg_sequences) > n_neg:
        random.seed(42)
        non_arg_sequences = random.sample(non_arg_sequences, n_neg)
    
    # 创建标签
    sequences = arg_sequences + non_arg_sequences
    labels = [1] * len(arg_sequences) + [0] * len(non_arg_sequences)
    
    logger.info(f"Final dataset: {len(arg_sequences)} positive, {len(non_arg_sequences)} negative")
    
    return sequences, labels


# ==================== 模型定义 ====================

class ESM2FeatureExtractor(nn.Module):
    """ESM-2 特征提取器"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", freeze_layers=4):
        super().__init__()
        
        try:
            from transformers import EsmModel, EsmTokenizer as HFTokenizer
            self.esm = EsmModel.from_pretrained(model_name)
            self.use_hf = True
            logger.info(f"Loaded ESM-2 model: {model_name}")
        except ImportError:
            logger.error("Please install transformers: pip install transformers")
            raise
        except Exception as e:
            logger.warning(f"Failed to load ESM-2 from HuggingFace: {e}")
            logger.warning("Falling back to local implementation")
            self.use_hf = False
            self.esm = None
        
        if self.use_hf:
            # 获取隐藏层维度
            self.hidden_size = self.esm.config.hidden_size
            
            # 冻结部分层
            if freeze_layers > 0:
                self._freeze_layers(freeze_layers)
                logger.info(f"Frozen first {freeze_layers} layers of ESM-2")
        else:
            self.hidden_size = 320  # esm2_t6_8M_UR50D 的隐藏层维度
    
    def _freeze_layers(self, n_layers):
        """冻结前 n 层"""
        # 冻结 embeddings
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结指定层数的 encoder
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        返回最后一层的隐藏状态
        output: (batch, seq_len, hidden_size)
        """
        if self.use_hf:
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
        else:
            # 备用方案：使用简单的 embedding
            raise NotImplementedError("Please install transformers library")


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, hidden_size)
        mask: (batch, seq_len) bool tensor, True 表示需要 mask 的位置
        """
        # 使用 <cls> token 作为 query
        cls_query = x[:, 0:1, :]  # (batch, 1, hidden_size)
        
        # Attention
        attn_output, attn_weights = self.attention(
            cls_query, x, x, 
            key_padding_mask=mask,
            need_weights=True
        )
        
        return attn_output.squeeze(1), attn_weights


class BiLSTMClassifier(nn.Module):
    """BiLSTM 分类器，处理 ESM-2 输出"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类器
        classifier_input = hidden_size * 2  # BiLSTM
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, input_size)
        lengths: 实际序列长度
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Global Pooling
        max_pool, _ = torch.max(lstm_out, dim=1)
        avg_pool = torch.mean(lstm_out, dim=1)
        features = (max_pool + avg_pool) / 2
        
        # 分类
        return self.classifier(self.dropout(features))


class ESM2BiLSTMModel(nn.Module):
    """
    完整的 ESM-2 + BiLSTM 模型
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # ESM-2 特征提取器
        self.esm = ESM2FeatureExtractor(config.ESM_MODEL_NAME, config.ESM_FREEZE_LAYERS)
        
        # Attention 池化 (可选)
        self.use_attention = True
        if self.use_attention:
            self.attention_pool = AttentionPooling(
                self.esm.hidden_size, 
                config.ATTENTION_HEADS
            )
            lstm_input_size = self.esm.hidden_size
        else:
            lstm_input_size = self.esm.hidden_size
        
        # BiLSTM 分类器
        self.classifier = BiLSTMClassifier(
            input_size=lstm_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT,
            num_classes=1
        )
    
    def forward(self, input_ids, lengths=None):
        """
        input_ids: (batch, seq_len)
        lengths: (batch,)
        """
        # ESM-2 特征提取
        # 创建 attention mask
        attention_mask = (input_ids != 1).long()  # 1 是 <pad> token
        
        esm_output = self.esm(input_ids, attention_mask)
        # esm_output: (batch, seq_len, hidden_size)
        
        # 可选：使用 Attention Pooling 或直接输入 BiLSTM
        if self.use_attention:
            # 使用 attention pooling 提取全局特征
            pooled, _ = self.attention_pool(esm_output, attention_mask == 0)
            # 扩展维度以输入 LSTM (batch, 1, hidden) -> 需要调整
            features = pooled.unsqueeze(1).expand(-1, esm_output.size(1), -1)
            lstm_input = features
        else:
            lstm_input = esm_output
        
        # BiLSTM 分类
        logits = self.classifier(lstm_input, lengths)
        
        return logits


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
        
        # 前向传播
        logits = model(input_ids, lengths)
        logits = logits.squeeze()
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            logits = model(input_ids, lengths)
            logits = logits.squeeze()
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'probs': all_probs,
        'preds': all_preds,
        'labels': all_labels
    }


def plot_results(history, eval_results, save_dir):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss 曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC 曲线
    fpr, tpr, _ = roc_curve(eval_results['labels'], eval_results['probs'])
    roc_auc = eval_results['roc_auc']
    axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'r--')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 混淆矩阵
    cm = confusion_matrix(eval_results['labels'], eval_results['preds'])
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_xticklabels(['Non-ARG', 'ARG'])
    axes[1, 1].set_yticklabels(['Non-ARG', 'ARG'])
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
    
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'training_results.pdf'))
    plt.close()


def main():
    """主训练流程"""
    config = Config()
    set_seed(config.SEED)
    
    # 创建目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Config: {config.__dict__}")
    
    # 准备数据
    sequences, labels = prepare_data(config)
    
    # 划分训练集/验证集
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=config.SEED, stratify=labels
    )
    
    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")
    
    # 创建 tokenizer 和 dataset
    tokenizer = ESMTokenizer()
    train_dataset = ARGDataset(train_seqs, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = ARGDataset(val_seqs, val_labels, tokenizer, config.MAX_LENGTH)
    
    # 创建 dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # ESM-2 可能会与多进程冲突
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
    model = ESM2BiLSTMModel(config).to(config.DEVICE)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 损失函数和优化器
    # 计算 pos_weight
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos).to(config.DEVICE)
    logger.info(f"Pos weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 分别设置学习率：ESM-2 使用较小学习率
    esm_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'esm' in name:
                esm_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': esm_params, 'lr': config.LR * 0.1},  # ESM-2 使用较小学习率
        {'params': other_params, 'lr': config.LR}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = os.path.join(config.SAVE_DIR, f'esm2_bilstm_binary_{timestamp}.pth')
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(config.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # 验证
        eval_results = evaluate(model, val_loader, criterion, config.DEVICE)
        val_loss = eval_results['loss']
        val_acc = eval_results['accuracy']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val Precision: {eval_results['precision']:.4f}, "
                   f"Recall: {eval_results['recall']:.4f}, "
                   f"F1: {eval_results['f1']:.4f}, "
                   f"AUC: {eval_results['roc_auc']:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'performance': eval_results,
            }, save_path)
            logger.info(f"✓ Best model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"✗ No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                logger.info("Early stopping triggered!")
                break
    
    # 最终评估和绘图
    logger.info("\n" + "=" * 60)
    logger.info("Training completed! Loading best model for final evaluation...")
    logger.info("=" * 60)
    
    checkpoint = torch.load(save_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate(model, val_loader, criterion, config.DEVICE)
    
    logger.info("\nFinal Evaluation Results:")
    logger.info(f"Accuracy:  {final_results['accuracy']:.4f}")
    logger.info(f"Precision: {final_results['precision']:.4f}")
    logger.info(f"Recall:    {final_results['recall']:.4f}")
    logger.info(f"F1 Score:  {final_results['f1']:.4f}")
    logger.info(f"ROC AUC:   {final_results['roc_auc']:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(final_results['labels'], final_results['preds'])}")
    
    # 绘制结果
    plot_results(history, final_results, config.FIG_DIR)
    logger.info(f"\nFigures saved to {config.FIG_DIR}")
    
    return save_path


if __name__ == "__main__":
    main()

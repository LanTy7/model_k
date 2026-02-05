#!/usr/bin/env python3
"""
ARG 二分类模型 v1.1 - 批量预测脚本
使用训练好的 ESM-2 + BiLSTM 模型进行预测
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
import argparse

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_train'))


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


class InferenceDataset(Dataset):
    """推理用数据集"""
    
    def __init__(self, fasta_file, tokenizer, max_length):
        self.records = list(SeqIO.parse(fasta_file, "fasta"))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        seq = str(self.records[idx].seq).upper()
        tokens = self.tokenizer.encode(seq, self.max_length)
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """自定义 batch 处理函数"""
    return torch.stack(batch)


def load_model(model_path, device):
    """加载训练好的模型"""
    from model_train.train import ESM2BiLSTMModel, Config
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    
    # 创建配置对象
    config = Config()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.DEVICE = device
    
    # 创建模型
    model = ESM2BiLSTMModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def predict_file(model, config, input_path, output_path, threshold=0.5, device='cuda'):
    """对单个 FASTA 文件进行预测"""
    
    tokenizer = ESMTokenizer()
    dataset = InferenceDataset(input_path, tokenizer, config.MAX_LENGTH)
    
    if len(dataset) == 0:
        return 0, []
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    arg_records = []
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            for i, prob in enumerate(probs):
                global_idx = batch_idx * dataloader.batch_size + i
                record = dataset.records[global_idx]
                
                result = {
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq),
                    'arg_probability': float(prob),
                    'is_arg': prob > threshold
                }
                results.append(result)
                
                if prob > threshold:
                    record.description += f" [ARG_prob={prob:.4f}]"
                    arg_records.append(record)
    
    # 保存预测的 ARG 序列
    if arg_records:
        SeqIO.write(arg_records, output_path, "fasta")
    
    return len(arg_records), results


def main():
    parser = argparse.ArgumentParser(description='ARG Binary Classification Predictor v1.1')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file or directory')
    parser.add_argument('--output', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device)
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    print(f"Model config: MAX_LENGTH={config.MAX_LENGTH}, BATCH_SIZE={config.BATCH_SIZE}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取输入文件
    if os.path.isdir(args.input):
        input_files = (glob.glob(os.path.join(args.input, "*.fasta")) + 
                      glob.glob(os.path.join(args.input, "*.faa")) +
                      glob.glob(os.path.join(args.input, "*.fa")))
    else:
        input_files = [args.input]
    
    print(f"Found {len(input_files)} files to process")
    
    # 批量预测
    all_results = []
    total_args = 0
    
    for input_file in tqdm(input_files, desc="Processing files"):
        basename = os.path.basename(input_file)
        name, ext = os.path.splitext(basename)
        output_file = os.path.join(args.output, f"{name}_predicted{ext}")
        
        n_args, results = predict_file(model, config, input_file, output_file, args.threshold, device)
        total_args += n_args
        
        # 添加文件名到结果
        for r in results:
            r['source_file'] = basename
        all_results.extend(results)
    
    # 保存完整结果 CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Prediction completed!")
    print(f"Total sequences: {len(df)}")
    print(f"Predicted ARGs: {total_args} ({total_args/len(df)*100:.2f}%)")
    print(f"Results saved to: {args.output}")
    print(f"CSV report: {csv_path}")
    
    # 显示概率分布
    print(f"\nProbability distribution:")
    print(df['arg_probability'].describe())
    
    return df


if __name__ == "__main__":
    main()

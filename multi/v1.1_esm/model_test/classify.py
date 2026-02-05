#!/usr/bin/env python3
"""
ARG 多分类模型 v1.1 - 批量分类脚本
使用训练好的 ESM-2 + BiLSTM 模型进行分类
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
    
    def __init__(self, records, tokenizer, max_length):
        self.records = records
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
    from model_train.train import ESM2BiLSTMClassifier, Config
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    class_names = checkpoint['class_names']
    
    # 创建配置对象
    config = Config()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.DEVICE = device
    
    # 创建模型
    num_classes = len(class_names)
    model = ESM2BiLSTMClassifier(config, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, class_names


def classify_batch(model, config, class_names, records, device='cuda'):
    """对一批记录进行分类"""
    
    if len(records) == 0:
        return []
    
    tokenizer = ESMTokenizer()
    dataset = InferenceDataset(records, tokenizer, config.MAX_LENGTH)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            for i in range(len(preds)):
                global_idx = batch_idx * dataloader.batch_size + i
                record = records[global_idx]
                pred_class = class_names[preds[i].item()]
                confidence = max_probs[i].item()
                
                # 获取前3个预测
                top3_probs, top3_indices = torch.topk(probs[i], k=min(3, len(class_names)))
                top3_predictions = [
                    {
                        'class': class_names[idx.item()],
                        'probability': prob.item()
                    }
                    for idx, prob in zip(top3_indices, top3_probs)
                ]
                
                result = {
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq),
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'top3_predictions': top3_predictions
                }
                results.append(result)
    
    return results


def classify_file(model, config, class_names, input_path, device='cuda'):
    """对单个 FASTA 文件进行分类"""
    records = list(SeqIO.parse(input_path, "fasta"))
    
    if len(records) == 0:
        return []
    
    results = classify_batch(model, config, class_names, records, device)
    
    # 添加源文件信息
    basename = os.path.basename(input_path)
    for r in results:
        r['source_file'] = basename
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ARG Multi-Class Classification v1.1')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file or directory')
    parser.add_argument('--output', type=str, default='./classification_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, 
                       help='Minimum confidence threshold for predictions')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.model}...")
    model, config, class_names = load_model(args.model, device)
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    print(f"Model config: MAX_LENGTH={config.MAX_LENGTH}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
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
    
    # 批量分类
    all_results = []
    
    for input_file in tqdm(input_files, desc="Processing files"):
        results = classify_file(model, config, class_names, input_file, device)
        all_results.extend(results)
    
    # 过滤低置信度预测
    if args.confidence_threshold > 0:
        filtered_results = [r for r in all_results if r['confidence'] >= args.confidence_threshold]
        print(f"Filtered {len(all_results) - len(filtered_results)} low-confidence predictions")
        all_results = filtered_results
    
    # 保存结果
    # 主 CSV (不包含 top3_predictions 详细信息)
    df_main = pd.DataFrame([
        {
            'id': r['id'],
            'source_file': r['source_file'],
            'predicted_class': r['predicted_class'],
            'confidence': r['confidence'],
            'sequence': r['sequence'][:100] + '...' if len(r['sequence']) > 100 else r['sequence']
        }
        for r in all_results
    ])
    
    csv_path = os.path.join(args.output, 'classification_results.csv')
    df_main.to_csv(csv_path, index=False)
    
    # 详细 JSON (包含 top3 预测)
    import json
    json_path = os.path.join(args.output, 'classification_results_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Classification completed!")
    print(f"Total sequences classified: {len(all_results)}")
    
    # 类别分布
    print(f"\nClass distribution:")
    class_dist = df_main['predicted_class'].value_counts()
    for class_name, count in class_dist.items():
        percentage = count / len(df_main) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    # 置信度分布
    print(f"\nConfidence distribution:")
    print(df_main['confidence'].describe())
    
    return df_main


if __name__ == "__main__":
    main()

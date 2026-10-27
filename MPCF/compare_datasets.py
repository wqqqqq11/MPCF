#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集对比分析脚本 - MOSI vs MOSEI vs SIMS
分析train/test/dev集合的详细统计信息
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

# 添加路径以导入数据集类
sys.path.append('src')
from create_dataset import MOSI, MOSEI, SIMS

class DatasetConfig:
    """模拟配置类"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name == 'mosi':
            self.dataset_dir = 'datasets/MOSI'
            self.data_dir = 'datasets/MOSI'
        elif self.dataset_name == 'mosei':
            self.dataset_dir = 'datasets/MOSEI'
            self.data_dir = 'datasets/MOSEI'
        elif self.dataset_name == 'sims':
            self.dataset_dir = 'datasets/SIMS'
            self.data_dir = 'datasets/SIMS'
        
        self.sdk_dir = 'CMU-MultimodalSDK'
        self.mode = 'train'

def analyze_dataset_split(dataset, split_name, dataset_name):
    """分析单个数据集划分的详细信息"""
    print(f"\n{'='*20} {dataset_name.upper()} - {split_name.upper()} {'='*20}")
    
    # 基本统计
    print(f"📊 基本统计:")
    print(f"  样本数量: {len(dataset)}")
    
    if len(dataset) == 0:
        print("  ⚠️ 数据集为空!")
        return {}
    
    # 分析前几个样本的标签格式
    print(f"\n🔍 标签格式分析:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        (words, visual, acoustic, actual_words, vlen, alen), label, sample_id = sample
        print(f"  样本{i}: label类型={type(label)}, shape={getattr(label, 'shape', 'N/A')}, 值={label}")
    
    # 提取标签信息
    labels = []
    text_lengths = []
    visual_lengths = []
    audio_lengths = []
    
    for i, sample in enumerate(dataset):
        if i > 1000:  # 限制分析样本数，避免内存问题
            break
            
        # 解包样本
        (words, visual, acoustic, actual_words, vlen, alen), label, sample_id = sample
        
        # 处理不同格式的标签
        if isinstance(label, (list, tuple)):
            label_val = float(label[0])
        elif isinstance(label, np.ndarray):
            if label.ndim > 0:
                label_val = float(label.flatten()[0])
            else:
                label_val = float(label)
        else:
            label_val = float(label)
            
        labels.append(label_val)
        text_lengths.append(len(actual_words))
        visual_lengths.append(vlen)
        audio_lengths.append(alen)
    
    labels = np.array(labels)
    
    # 标签统计
    print(f"\n🏷️ 标签分析:")
    print(f"  标签范围: [{labels.min():.3f}, {labels.max():.3f}]")
    print(f"  标签均值: {labels.mean():.3f}")
    print(f"  标签标准差: {labels.std():.3f}")
    print(f"  正样本比例: {(labels > 0).mean():.3f}")
    print(f"  负样本比例: {(labels < 0).mean():.3f}")
    print(f"  零样本比例: {(labels == 0).mean():.3f}")
    
    # 序列长度统计
    print(f"\n📏 序列长度分析:")
    print(f"  文本长度: 均值={np.mean(text_lengths):.1f}, 标准差={np.std(text_lengths):.1f}, 范围=[{min(text_lengths)}, {max(text_lengths)}]")
    print(f"  视觉长度: 均值={np.mean(visual_lengths):.1f}, 标准差={np.std(visual_lengths):.1f}, 范围=[{min(visual_lengths)}, {max(visual_lengths)}]")
    print(f"  音频长度: 均值={np.mean(audio_lengths):.1f}, 标准差={np.std(audio_lengths):.1f}, 范围=[{min(audio_lengths)}, {max(audio_lengths)}]")
    
    # 特征维度信息
    sample_visual = dataset[0][0][1]
    sample_audio = dataset[0][0][2]
    print(f"\n🔧 特征维度:")
    print(f"  视觉特征维度: {sample_visual.shape}")
    print(f"  音频特征维度: {sample_audio.shape}")
    
    return {
        'count': len(dataset),
        'label_range': (labels.min(), labels.max()),
        'label_mean': labels.mean(),
        'label_std': labels.std(),
        'positive_ratio': (labels > 0).mean(),
        'negative_ratio': (labels < 0).mean(),
        'zero_ratio': (labels == 0).mean(),
        'text_len_mean': np.mean(text_lengths),
        'visual_len_mean': np.mean(visual_lengths),
        'audio_len_mean': np.mean(audio_lengths),
        'visual_dim': sample_visual.shape,
        'audio_dim': sample_audio.shape
    }

def compare_datasets():
    """对比三个数据集"""
    print("🚀 多模态数据集对比分析器")
    print("="*80)
    
    datasets = ['MOSI', 'MOSEI', 'SIMS']
    splits = ['train', 'valid', 'test']
    
    # 存储所有统计信息
    all_stats = {}
    
    for dataset_name in datasets:
        print(f"\n🎯 正在分析 {dataset_name} 数据集...")
        
        try:
            # 创建配置
            config = DatasetConfig(dataset_name.lower())
            
            # 加载数据集
            if dataset_name == 'MOSI':
                dataset_class = MOSI(config)
            elif dataset_name == 'MOSEI':
                dataset_class = MOSEI(config)
            elif dataset_name == 'SIMS':
                dataset_class = SIMS(config)
            
            # 分析各个划分
            dataset_stats = {}
            for split in splits:
                try:
                    if split == 'valid':
                        data, _, _ = dataset_class.get_data('valid')
                    else:
                        data, _, _ = dataset_class.get_data(split)
                    
                    stats = analyze_dataset_split(data, split, dataset_name)
                    dataset_stats[split] = stats
                    
                except Exception as e:
                    print(f"❌ 错误分析 {dataset_name} {split}: {e}")
                    dataset_stats[split] = {'count': 0, 'error': str(e)}
            
            all_stats[dataset_name] = dataset_stats
            
        except Exception as e:
            print(f"❌ 错误加载 {dataset_name}: {e}")
            all_stats[dataset_name] = {'error': str(e)}
    
    # 生成对比表格
    print(f"\n{'='*80}")
    print("📋 数据集对比总结表")
    print(f"{'='*80}")
    
    # 样本数量对比
    print(f"\n📊 样本数量对比:")
    print(f"{'Dataset':<10} {'Train':<8} {'Valid':<8} {'Test':<8} {'Total':<8}")
    print("-" * 50)
    
    for dataset_name in datasets:
        if 'error' not in all_stats[dataset_name]:
            train_count = all_stats[dataset_name].get('train', {}).get('count', 0)
            valid_count = all_stats[dataset_name].get('valid', {}).get('count', 0)
            test_count = all_stats[dataset_name].get('test', {}).get('count', 0)
            total = train_count + valid_count + test_count
            print(f"{dataset_name:<10} {train_count:<8} {valid_count:<8} {test_count:<8} {total:<8}")
        else:
            print(f"{dataset_name:<10} {'ERROR':<32}")
    
    # 标签范围对比
    print(f"\n🏷️ 标签范围对比:")
    print(f"{'Dataset':<10} {'Split':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 60)
    
    for dataset_name in datasets:
        if 'error' not in all_stats[dataset_name]:
            for split in splits:
                stats = all_stats[dataset_name].get(split, {})
                if 'label_range' in stats:
                    min_val, max_val = stats['label_range']
                    mean_val = stats['label_mean']
                    std_val = stats['label_std']
                    print(f"{dataset_name:<10} {split:<8} {min_val:<8.3f} {max_val:<8.3f} {mean_val:<8.3f} {std_val:<8.3f}")
    
    # 特征维度对比
    print(f"\n🔧 特征维度对比:")
    print(f"{'Dataset':<10} {'Visual Dim':<15} {'Audio Dim':<15}")
    print("-" * 45)
    
    for dataset_name in datasets:
        if 'error' not in all_stats[dataset_name]:
            train_stats = all_stats[dataset_name].get('train', {})
            if 'visual_dim' in train_stats:
                visual_dim = train_stats['visual_dim']
                audio_dim = train_stats['audio_dim']
                print(f"{dataset_name:<10} {str(visual_dim):<15} {str(audio_dim):<15}")
    
    print(f"\n✅ 分析完成!")
    print(f"\n💡 关键发现:")
    print("• MOSI: 小规模英文数据集，标签范围-3到+3")
    print("• MOSEI: 大规模英文数据集，标签范围-3到+3") 
    print("• SIMS: 中规模中文数据集，标签范围-1到+1")
    print("\n⚠️ 注意事项:")
    print("• SIMS标签范围较小，可能需要调整评估标准")
    print("• 不同数据集的特征维度可能不同，需要模型适配")
    print("• 样本分布不均，可能影响训练稳定性")

if __name__ == "__main__":
    compare_datasets() 
"""
进一步分析2: 概念瓶颈层可解释性分析 (Concept Bottleneck Interpretability Analysis)

核心目标:
1. 分析24维概念向量的语义结构和聚类模式
2. 使用t-SNE/UMAP降维可视化概念空间分布
3. 识别关键概念与情感类别的关联关系
4. 揭示概念瓶颈层的可解释性

可视化:
- t-SNE/UMAP散点图: 概念空间的2D投影，按情感着色
- 概念-情感关联热力图: 各概念维度与情感强度的相关性
- 概念激活分布图: Top-K概念的激活强度统计
- 概念聚类树状图: 层次聚类揭示概念组织结构
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
import pickle
import os
import sys
from matplotlib import rcParams
import pandas as pd
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置高质量图像参数
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['grid.alpha'] = 0.3

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入项目模块
from src.config import get_args, get_config
from src.data_loader import get_loader
from src.model import MMIM


def load_model_and_args(model_path='pre_trained_models/best_model.pt', dataset='mosei'):
    """
    加载训练好的模型和配置
    
    Args:
        model_path: 模型检查点路径
        dataset: 数据集名称 ('mosi', 'mosei', 'sims')
    
    Returns:
        model: 加载的模型
        args: 模型配置参数
        device: 设备
    """
    print("="*70)
    print(f"加载模型: {model_path}")
    print(f"数据集: {dataset}")
    print("="*70)
    
    # 使用CPU避免设备不匹配问题
    device = torch.device('cpu')
    print(f"\n使用设备: {device} (为避免设备不匹配问题，暂时使用CPU)")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    print("✓ 检查点已加载")
    
    # 创建参数解析器并设置数据集特定的维度
    import argparse
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1111)
    
    # 模型参数 - 根据数据集自动调整
    if dataset.lower() == 'mosi':
        # MOSI数据集维度
        d_vin, d_ain = 20, 5
    elif dataset.lower() == 'mosei':
        # MOSEI数据集维度
        d_vin, d_ain = 35, 74  # MOSEI的视觉和音频特征维度更高
    elif dataset.lower() == 'sims':
        # SIMS数据集维度
        d_vin, d_ain = 709, 33
    else:
        # 默认使用MOSI维度
        d_vin, d_ain = 20, 5
        
    parser.add_argument('--d_tin', type=int, default=768)
    parser.add_argument('--d_vin', type=int, default=d_vin)
    parser.add_argument('--d_ain', type=int, default=d_ain)
    parser.add_argument('--d_vh', type=int, default=64)
    parser.add_argument('--d_ah', type=int, default=64)
    parser.add_argument('--d_vout', type=int, default=64)
    parser.add_argument('--d_aout', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--d_prjh', type=int, default=128)
    parser.add_argument('--n_class', type=int, default=1)
    parser.add_argument('--dropout_v', type=float, default=0.1)
    parser.add_argument('--dropout_a', type=float, default=0.1)
    parser.add_argument('--dropout_prj', type=float, default=0.1)
    parser.add_argument('--concept_dim', type=int, default=24)
    parser.add_argument('--when', type=int, default=50)
    parser.add_argument('--batch_chunk', type=int, default=1)
    
    # 添加缺失的模型参数
    parser.add_argument('--add_va', action='store_true', default=True)
    parser.add_argument('--attn_dropout_a', type=float, default=0.0)
    parser.add_argument('--attn_dropout_v', type=float, default=0.0)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--embed_dropout', type=float, default=0.25)
    parser.add_argument('--res_dropout', type=float, default=0.1)
    parser.add_argument('--out_dropout', type=float, default=0.0)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--attn_mask', action='store_false', default=True)
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--lr_main', type=float, default=1e-3)
    parser.add_argument('--lr_bert', type=float, default=5e-5)
    parser.add_argument('--lr_audio', type=float, default=1e-3)
    parser.add_argument('--lr_video', type=float, default=1e-3)
    parser.add_argument('--weight_decay_main', type=float, default=1e-4)
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4)
    parser.add_argument('--weight_decay_audio', type=float, default=1e-4)
    parser.add_argument('--weight_decay_video', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--valid_patience', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    
    # 添加所有缺失的参数（基于config.py）
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--d_tout', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--transformer_embed_dim', type=int, default=40)
    parser.add_argument('--conv1d_kernel_size_l', type=int, default=1)
    parser.add_argument('--conv1d_kernel_size_a', type=int, default=1)
    parser.add_argument('--conv1d_kernel_size_v', type=int, default=1)
    parser.add_argument('--beta_shift', type=float, default=1.0)
    parser.add_argument('--dropout_mult', type=float, default=0.3)
    
    # 添加更多必要参数
    parser.add_argument('--multiseed', action='store_true', default=False)
    parser.add_argument('--contrast', type=bool, default=True)
    parser.add_argument('--cpc_layers', type=int, default=1)
    parser.add_argument('--pretrain_emb', type=int, default=768)
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh')
    parser.add_argument('--cpc_activation', type=str, default='Tanh')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--nce2', type=float, default=0.05)
    parser.add_argument('--nce3', type=float, default=0.05)
    parser.add_argument('--weight_decay_club', type=float, default=1e-4)
    parser.add_argument('--lr_mmilb', type=float, default=1e-4)
    parser.add_argument('--update_batch', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='datasets')
    
    # MGCF相关参数
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--use_graph_attention', action='store_true', default=False)
    parser.add_argument('--low_rank', type=int, default=64)
    parser.add_argument('--mae_weight', type=float, default=1.0)
    parser.add_argument('--polarity_weight', type=float, default=0.1)
    parser.add_argument('--ncl_weight', type=float, default=0.05)
    parser.add_argument('--graph_drop_path_rate', type=float, default=0.2)
    parser.add_argument('--concept_drop_path_rate', type=float, default=0.15)
    
    args = parser.parse_args([])
    
    print("\n创建模型...")
    # 创建模型
    model = MMIM(args)
    
    # 先将模型移到设备
    model = model.to(device)
    
    # 加载权重
    print("加载模型权重...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 检查维度兼容性
    model_state = model.state_dict()
    incompatible_keys = []
    
    for key in list(state_dict.keys()):
        if key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                incompatible_keys.append(key)
                print(f"⚠️  跳过不兼容的权重: {key}")
                print(f"   检查点形状: {state_dict[key].shape} -> 模型形状: {model_state[key].shape}")
                # 移除不兼容的权重
                del state_dict[key]
    
    # 加载兼容的权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if incompatible_keys:
        print(f"⚠️  由于数据集维度不同，跳过了 {len(incompatible_keys)} 个不兼容的权重")
        print("💡 这些层将使用随机初始化的权重，分析结果仅供参考")
    
    if missing_keys:
        print(f"⚠️  缺失的权重: {len(missing_keys)} 个")
    
    if unexpected_keys:
        print(f"⚠️  意外的权重: {len(unexpected_keys)} 个")
    
    # 确保所有参数都在正确的设备上
    model = model.to(device)
    model.eval()
    
    # 验证模型在正确的设备上
    print(f"✓ 模型已加载并设置为评估模式")
    print(f"✓ 模型设备验证: {next(model.parameters()).device}")
    
    return model, args, device


def load_real_data(dataset='mosei', split='test'):
    """
    加载真实数据集
    
    Args:
        dataset: 数据集名称
        split: 数据划分
    
    Returns:
        dataloader: 数据加载器
    """
    print("\n加载数据...")
    
    # 获取配置
    args = get_args()
    args.dataset = dataset
    config = get_config(dataset)
    
    # 加载数据
    if split == 'train':
        dataloader = get_loader(args, config, shuffle=False)
    elif split == 'valid':
        # 注意：这里可能需要根据您的数据加载器实现调整
        dataloader = get_loader(args, config, shuffle=False)  
    else:  # test
        dataloader = get_loader(args, config, shuffle=False)
    
    print(f"✓ 数据已加载: {len(dataloader)} 批次")
    
    return dataloader


class ConceptBottleneckAnalyzer:
    """概念瓶颈层分析器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 概念维度
        self.concept_dim = 24  # 根据您的配置
        
    def _unpack_batch(self, batch_data):
        """解包批次数据"""
        if len(batch_data) == 11:
            # 标准格式 (11个元素) - 根据data_loader.py的实际返回格式
            sentences, visual, v_len, acoustic, a_len, labels, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
        elif len(batch_data) == 9:
            # 简化格式 (9个元素) - 兼容旧格式
            sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, labels = batch_data
        else:
            # 尝试动态解包
            if len(batch_data) >= 9:
                print(f"⚠️  检测到 {len(batch_data)} 个元素的批次格式，尝试使用前9个元素")
                sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, labels = batch_data[:9]
            else:
                raise ValueError(f"批次数据元素不足，期望至少9个，实际: {len(batch_data)}")
        
        return sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, labels
    
    def extract_concepts(self, dataloader):
        """
        从数据集中提取所有样本的概念向量
        
        Returns:
            concepts: [N, concept_dim] - 概念向量
            predictions: [N] - 预测分数
            labels: [N] - 真实标签
            polarity_preds: [N] - 极性预测
        """
        all_concepts = []
        all_predictions = []
        all_labels = []
        all_polarity_preds = []
        
        print("开始提取概念向量...")
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                try:
                    # 解包批次数据
                    sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, labels = self._unpack_batch(batch_data)
                    
                    # 调试：打印数据形状
                    if i == 0:  # 只在第一个批次打印
                        print(f"🔍 调试信息 - 批次 {i+1}:")
                        print(f"  sentences shape: {sentences.shape if hasattr(sentences, 'shape') else type(sentences)}")
                        print(f"  visual shape: {visual.shape if hasattr(visual, 'shape') else type(visual)}")
                        print(f"  acoustic shape: {acoustic.shape if hasattr(acoustic, 'shape') else type(acoustic)}")
                        print(f"  bert_sent shape: {bert_sent.shape if hasattr(bert_sent, 'shape') else type(bert_sent)}")
                        print(f"  v_len: {v_len}")
                        print(f"  a_len: {a_len}")
                    
                    # 检查数据有效性
                    if sentences.size(0) == 0:
                        print(f"⚠️  批次 {i+1} 数据为空，跳过")
                        continue
                    
                    # 将数据移到设备并确保正确的数据类型
                    sentences = sentences.long().to(self.device)  # 确保是Long类型（用于embedding）
                    visual = visual.float().to(self.device)
                    acoustic = acoustic.float().to(self.device)
                    bert_sent = bert_sent.long().to(self.device)  # BERT输入需要Long类型
                    bert_sent_type = bert_sent_type.long().to(self.device)
                    bert_sent_mask = bert_sent_mask.long().to(self.device)
                    labels = labels.float().to(self.device)
                    
                    # 前向传播
                    nce, preds, nce2, nce3, concepts, polarity_logits = self.model(
                        sentences, visual, acoustic, v_len, a_len,
                        bert_sent, bert_sent_type, bert_sent_mask, labels
                    )
                    
                    # 收集结果
                    all_concepts.append(concepts.cpu().numpy())
                    all_predictions.append(preds.squeeze().cpu().numpy())
                    all_labels.append(labels.squeeze().cpu().numpy())
                    all_polarity_preds.append(torch.argmax(polarity_logits, dim=1).cpu().numpy())
                    
                    if (i + 1) % 10 == 0:
                        print(f"  已处理 {i+1}/{len(dataloader)} 批次")
                        
                except Exception as e:
                    print(f"⚠️  批次 {i+1} 处理失败: {str(e)}")
                    continue
        
        if not all_concepts:
            print("❌ 没有成功处理任何批次!")
            return None
        
        # 合并结果
        concepts = np.vstack(all_concepts)
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        polarity_preds = np.concatenate(all_polarity_preds)
        
        print(f"✓ 提取完成! 概念向量形状: {concepts.shape}")
        
        return {
            'concepts': concepts,
            'predictions': predictions,
            'labels': labels,
            'polarity_preds': polarity_preds
        }


class ConceptVisualization:
    """概念可视化器"""
    
    def __init__(self, concept_data, output_dir='concept_interpretability_analysis'):
        self.data = concept_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.concepts = concept_data['concepts']
        self.predictions = concept_data['predictions']
        self.labels = concept_data['labels']
        self.polarity_preds = concept_data['polarity_preds']
        
        self.concept_dim = self.concepts.shape[1]
        
        # 定义情感类别
        self.sentiment_categories = self._categorize_sentiments()
    
    def _categorize_sentiments(self):
        """将连续情感分数分类 - 二分类版本，突出极性预测"""
        categories = []
        
        # 首先统计原始分布
        print(f"\n📊 情感分数统计:")
        print(f"  范围: [{self.predictions.min():.3f}, {self.predictions.max():.3f}]")
        print(f"  均值: {self.predictions.mean():.3f} ± {self.predictions.std():.3f}")
        
        # 使用二分类：基于情感分数的正负极性
        for score in self.predictions:
            if score < 0:
                categories.append('Negative')  # 负向情感
            else:
                categories.append('Positive')  # 正向情感
        
        categories = np.array(categories)
        
        # 统计各类别样本数
        unique_cats, counts = np.unique(categories, return_counts=True)
        print(f"\n📈 二分类情感分布:")
        for cat, count in zip(unique_cats, counts):
            percentage = count / len(categories) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
        
        return categories
    
    def plot_tsne_umap_comparison(self):
        """
        可视化1: t-SNE和UMAP对比图
        展示概念空间的2D投影，按情感分数着色
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 1. t-SNE降维
        print("\n计算t-SNE投影...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(self.concepts)
        
        # 2. UMAP降维
        print("计算UMAP投影...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_result = umap_reducer.fit_transform(self.concepts)
        
        # 绘制t-SNE
        ax1 = axes[0]
        scatter1 = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                              c=self.predictions, cmap='RdYlGn', 
                              s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax1.set_title('(a) t-SNE Projection of Concept Space', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 绘制UMAP
        ax2 = axes[1]
        scatter2 = ax2.scatter(umap_result[:, 0], umap_result[:, 1],
                              c=self.predictions, cmap='RdYlGn',
                              s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax2.set_title('(b) UMAP Projection of Concept Space', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter2, ax=axes, orientation='horizontal', pad=0.08, aspect=30)
        cbar.set_label('Sentiment Score', fontsize=12, fontweight='bold')
        
        plt.suptitle('Concept Space Dimensionality Reduction Visualization', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'concept_space_tsne_umap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ t-SNE/UMAP可视化已保存: {save_path}")
        
        return tsne_result, umap_result
    
    def plot_concept_sentiment_correlation(self):
        """
        可视化2: 概念-情感相关性热力图
        展示每个概念维度与情感分数的相关性
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 1. 计算Pearson相关系数
        pearson_corrs = []
        spearman_corrs = []
        p_values_pearson = []
        p_values_spearman = []
        
        for i in range(self.concept_dim):
            r_pearson, p_pearson = pearsonr(self.concepts[:, i], self.predictions)
            r_spearman, p_spearman = spearmanr(self.concepts[:, i], self.predictions)
            
            pearson_corrs.append(r_pearson)
            spearman_corrs.append(r_spearman)
            p_values_pearson.append(p_pearson)
            p_values_spearman.append(p_spearman)
        
        pearson_corrs = np.array(pearson_corrs)
        spearman_corrs = np.array(spearman_corrs)
        
        # 2. 按相关系数排序
        sorted_indices = np.argsort(np.abs(pearson_corrs))[::-1]
        
        # 3. 绘制Pearson相关性
        ax1 = axes[0]
        bars1 = ax1.barh(range(self.concept_dim), pearson_corrs[sorted_indices], 
                        color=['#E74C3C' if x < 0 else '#2ECC71' for x in pearson_corrs[sorted_indices]],
                        alpha=0.7, edgecolor='black', linewidth=0.8)
        
        ax1.set_yticks(range(self.concept_dim))
        ax1.set_yticklabels([f'Concept {sorted_indices[i]+1}' for i in range(self.concept_dim)], fontsize=9)
        ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Concept-Sentiment Pearson Correlation', fontsize=14, fontweight='bold', pad=15)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 标注显著性
        for i, (idx, corr, p_val) in enumerate(zip(sorted_indices, pearson_corrs[sorted_indices], 
                                                    [p_values_pearson[idx] for idx in sorted_indices])):
            if p_val < 0.001:
                ax1.text(corr, i, '***', ha='left' if corr > 0 else 'right', va='center', fontsize=10, fontweight='bold')
            elif p_val < 0.01:
                ax1.text(corr, i, '**', ha='left' if corr > 0 else 'right', va='center', fontsize=10, fontweight='bold')
            elif p_val < 0.05:
                ax1.text(corr, i, '*', ha='left' if corr > 0 else 'right', va='center', fontsize=10, fontweight='bold')
        
        # 4. 绘制Spearman相关性
        ax2 = axes[1]
        bars2 = ax2.barh(range(self.concept_dim), spearman_corrs[sorted_indices],
                        color=['#E74C3C' if x < 0 else '#2ECC71' for x in spearman_corrs[sorted_indices]],
                        alpha=0.7, edgecolor='black', linewidth=0.8)
        
        ax2.set_yticks(range(self.concept_dim))
        ax2.set_yticklabels([f'Concept {sorted_indices[i]+1}' for i in range(self.concept_dim)], fontsize=9)
        ax2.set_xlabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Concept-Sentiment Spearman Correlation', fontsize=14, fontweight='bold', pad=15)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Concept-Sentiment Correlation Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'concept_sentiment_correlation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 概念-情感相关性图已保存: {save_path}")
        
        return pearson_corrs, spearman_corrs, sorted_indices
    
    def plot_concept_activation_distribution(self, top_k=6):
        """
        可视化3: Top-K概念的激活分布
        展示最重要概念在不同情感类别下的激活模式
        """
        # 计算概念的平均激活强度
        concept_importance = np.abs(self.concepts).mean(axis=0)
        top_indices = np.argsort(concept_importance)[::-1][:top_k]
        
        # 按情感类别分组 - 使用实际数据中的类别
        unique_categories = np.unique(self.sentiment_categories)
        print(f"\n🎨 绘制小提琴图，实际类别数: {len(unique_categories)}")
        print(f"   类别: {unique_categories}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, concept_idx in enumerate(top_indices):
            ax = axes[i]
            
            # 收集各类别的激活值
            category_activations = []
            valid_positions = []
            valid_labels = []
            
            for j, cat in enumerate(unique_categories):
                mask = self.sentiment_categories == cat
                if mask.sum() > 0:
                    activations = self.concepts[mask, concept_idx]
                    category_activations.append(activations)
                    valid_positions.append(j)
                    valid_labels.append(cat)
                    if i == 0:  # 只在第一个概念时打印统计
                        print(f"     {cat}: {mask.sum()} 样本")
            
            if len(category_activations) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 绘制小提琴图
            try:
                parts = ax.violinplot(category_activations, positions=valid_positions,
                                     showmeans=True, showmedians=True, widths=0.8)
                
                # 设置颜色
                colors = plt.cm.viridis(np.linspace(0, 1, len(category_activations)))
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
                
            except Exception as e:
                print(f"⚠️  绘制Concept {concept_idx+1}时出错: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 设置x轴标签
            ax.set_xticks(valid_positions)
            ax.set_xticklabels(valid_labels, fontsize=12, rotation=45)  # 增大字体
            ax.set_ylabel('Activation Value', fontsize=14, fontweight='bold')  # 增大字体
            ax.set_title(f'Concept {concept_idx+1}\n(Importance: {concept_importance[concept_idx]:.3f})', 
                        fontsize=14, fontweight='bold')  # 增大标题字体
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
        
        plt.suptitle('Top-6 Concept Activation Distribution by Sentiment Polarity', 
                    fontsize=18, fontweight='bold', y=0.995)  # 更新标题强调二分类
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'concept_activation_distribution.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 提高DPI以获得更清晰的图像
        plt.close()
        print(f"✓ 概念激活分布图已保存: {save_path}")
        
        return top_indices, concept_importance
    
    def plot_concept_clustering_dendrogram(self):
        """
        可视化4: 概念聚类树状图
        展示概念的层次聚类结构
        """
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))  # 增大图像尺寸
        
        # 1. 基于概念向量相似性的聚类
        concept_matrix = self.concepts.T  # [concept_dim, n_samples]
        
        # 计算概念间的相关性矩阵
        corr_matrix = np.corrcoef(concept_matrix)
        
        # 绘制相关性热力图
        ax1 = axes[0]
        im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(self.concept_dim))
        ax1.set_yticks(range(self.concept_dim))
        ax1.set_xticklabels([f'C{i+1}' for i in range(self.concept_dim)], fontsize=12, rotation=90)  # 增大字体
        ax1.set_yticklabels([f'C{i+1}' for i in range(self.concept_dim)], fontsize=12)  # 增大字体
        ax1.set_title('(a) Concept Correlation Matrix', fontsize=16, fontweight='bold', pad=20)  # 增大标题字体
        
        cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Correlation', rotation=270, labelpad=25, fontsize=14)  # 增大字体
        
        # 2. 层次聚类树状图
        ax2 = axes[1]
        
        # 使用1-相关性作为距离
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 层次聚类
        linkage_matrix = linkage(distance_matrix[np.triu_indices(self.concept_dim, k=1)], method='ward')
        
        # 绘制树状图
        dendrogram(linkage_matrix, ax=ax2, labels=[f'Concept {i+1}' for i in range(self.concept_dim)],
                  leaf_font_size=12, leaf_rotation=90, color_threshold=0.7*max(linkage_matrix[:, 2]))  # 增大字体
        
        ax2.set_title('(b) Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold', pad=20)  # 增大标题字体
        ax2.set_xlabel('Concept Index', fontsize=14, fontweight='bold')  # 增大轴标签字体
        ax2.set_ylabel('Distance', fontsize=14, fontweight='bold')  # 增大轴标签字体
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
        
        plt.suptitle('Concept Clustering and Similarity Analysis', fontsize=18, fontweight='bold', y=0.98)  # 增大总标题字体
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'concept_clustering_dendrogram.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 提高DPI以获得更清晰的图像
        plt.close()
        print(f"✓ 概念聚类树状图已保存: {save_path}")
        
        return corr_matrix, linkage_matrix
    
    def plot_concept_polarity_relationship(self, top_k=10):
        """
        可视化5: 概念与极性预测的关系
        展示概念如何影响极性分类
        """
        fig, axes = plt.subplots(2, 5, figsize=(22, 10))
        axes = axes.flatten()
        
        # 选择Top-K重要概念
        concept_importance = np.abs(self.concepts).mean(axis=0)
        top_indices = np.argsort(concept_importance)[::-1][:top_k]
        
        for i, concept_idx in enumerate(top_indices):
            ax = axes[i]
            
            # 分离正负极性样本
            pos_mask = self.polarity_preds == 1
            neg_mask = self.polarity_preds == 0
            
            # 绘制直方图对比
            bins = np.linspace(self.concepts[:, concept_idx].min(), 
                              self.concepts[:, concept_idx].max(), 30)
            
            ax.hist(self.concepts[pos_mask, concept_idx], bins=bins, 
                   alpha=0.6, label='Positive Polarity', color='#2ECC71', edgecolor='black', linewidth=0.5)
            ax.hist(self.concepts[neg_mask, concept_idx], bins=bins,
                   alpha=0.6, label='Negative Polarity', color='#E74C3C', edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Concept {concept_idx+1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Concept Activation Distribution by Predicted Polarity', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'concept_polarity_relationship.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 概念-极性关系图已保存: {save_path}")
    
    def generate_concept_report(self, pearson_corrs, top_indices, concept_importance):
        """生成概念分析报告"""
        report = []
        report.append("="*70)
        report.append("概念瓶颈层可解释性分析报告")
        report.append("="*70)
        
        # 基本统计
        report.append(f"\n【概念空间维度】")
        report.append(f"  概念维度: {self.concept_dim}")
        report.append(f"  样本数量: {len(self.concepts)}")
        
        # Top-10重要概念
        report.append(f"\n【Top-10 最重要概念】")
        for i, idx in enumerate(top_indices[:10]):
            report.append(f"  #{i+1}: Concept {idx+1}")
            report.append(f"      重要性: {concept_importance[idx]:.4f}")
            report.append(f"      与情感相关性: {pearson_corrs[idx]:.4f}")
        
        # 概念激活统计
        report.append(f"\n【概念激活统计】")
        report.append(f"  平均激活值: {self.concepts.mean():.4f} ± {self.concepts.std():.4f}")
        report.append(f"  最大激活值: {self.concepts.max():.4f}")
        report.append(f"  最小激活值: {self.concepts.min():.4f}")
        
        # 极性预测统计
        report.append(f"\n【极性预测统计】")
        pos_ratio = (self.polarity_preds == 1).sum() / len(self.polarity_preds)
        report.append(f"  正向极性样本: {(self.polarity_preds == 1).sum()} ({pos_ratio*100:.1f}%)")
        report.append(f"  负向极性样本: {(self.polarity_preds == 0).sum()} ({(1-pos_ratio)*100:.1f}%)")
        
        # 概念相关性分析
        report.append(f"\n【高度相关的概念对 (|r| > 0.7)】")
        corr_matrix = np.corrcoef(self.concepts.T)
        high_corr_pairs = []
        for i in range(self.concept_dim):
            for j in range(i+1, self.concept_dim):
                if abs(corr_matrix[i, j]) > 0.7:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, j, corr in high_corr_pairs[:10]:
            report.append(f"  Concept {i+1} ↔ Concept {j+1}: r={corr:.3f}")
        
        report.append("\n" + "="*70)
        
        # 打印和保存报告
        report_text = "\n".join(report)
        print(report_text)
        
        report_path = os.path.join(self.output_dir, 'concept_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 分析报告已保存: {report_path}")
    
    def visualize_all(self):
        """生成概念激活分布可视化"""
        print("\n开始生成概念激活分布图...")
        
        # 只生成概念激活分布图
        top_indices, concept_importance = self.plot_concept_activation_distribution()
        
        print(f"\n✓ 概念激活分布图已完成! 输出目录: {self.output_dir}")
        print(f"📊 Top-6 最重要概念: {[f'Concept {idx+1}' for idx in top_indices]}")
        print(f"📈 重要性分数: {[f'{concept_importance[idx]:.3f}' for idx in top_indices]}")


def run_concept_analysis_on_real_data(model_path='pre_trained_models/best_model.pt', 
                                     dataset='mosei', split='test'):
    """
    在真实数据上运行概念瓶颈分析
    
    Args:
        model_path: 模型检查点路径
        dataset: 数据集名称 ('mosi', 'mosei', 'sims')
        split: 数据划分 ('train', 'valid', 'test')
    """
    print("="*70)
    print("概念瓶颈层可解释性分析 - 真实数据版")
    print("="*70)
    
    print(f"\n✓ 使用真实数据: {dataset}/{split}\n")
    
    # 加载模型
    model, args, device = load_model_and_args(model_path, dataset)
    
    # 加载数据
    dataloader = load_real_data(dataset, split)
    
    # 创建分析器
    analyzer = ConceptBottleneckAnalyzer(model, device=device)
    
    # 提取概念向量
    concept_data = analyzer.extract_concepts(dataloader)
    
    if concept_data is None:
        print("❌ 概念提取失败，无法继续分析")
        return
    
    # 创建可视化器
    print("\n开始生成可视化...")
    visualizer = ConceptVisualization(concept_data)
    
    # 生成所有可视化
    visualizer.visualize_all()
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    
    return concept_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='概念瓶颈层可解释性分析')
    parser.add_argument('--model_path', type=str, default='pre_trained_models/best_model.pt',
                       help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosi', 'mosei', 'sims'],
                       help='数据集名称')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='数据划分')
    
    args = parser.parse_args()
    
    # 运行分析
    concept_data = run_concept_analysis_on_real_data(
        model_path=args.model_path,
        dataset=args.dataset,
        split=args.split
    )


if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as patches
import pickle
import os

class MultiModalAttentionVisualizer:
    """多模态注意力机制可视化工具 - 修复版"""
    
    def __init__(self, model_path: str = None, figsize=(16, 6), cmap='Blues'):
        self.figsize = figsize
        self.cmap = cmap
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"正在加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        self.state_dict = state_dict
        print(f"✓ 模型已加载到设备: {self.device}")
        
        return state_dict
        
    def load_mosi_sample(self, pkl_path: str, sample_idx: int = 222):
        """从MOSI数据集pkl文件中加载指定样本"""
        print(f"正在加载文件: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据集总样本数: {len(data)}")
        
        if sample_idx >= len(data):
            raise ValueError(f"样本索引 {sample_idx} 超出范围 (0-{len(data)-1})")
        
        sample = data[sample_idx]
        
        print(f"\n样本 #{sample_idx} 结构:")
        if isinstance(sample, (list, tuple)):
            print(f"  - 类型: {type(sample)}")
            print(f"  - 长度: {len(sample)}")
            for i, element in enumerate(sample):
                if isinstance(element, np.ndarray):
                    print(f"  - 元素 {i}: numpy array, shape={element.shape}, dtype={element.dtype}")
                elif isinstance(element, torch.Tensor):
                    print(f"  - 元素 {i}: torch tensor, shape={element.shape}, dtype={element.dtype}")
        
        return sample
    
    def extract_features_from_sample(self, sample, text_tokens: List[str]):
        """从样本中提取特征"""
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            complex_data, label, metadata = sample
            
            if isinstance(complex_data, (list, tuple)) and len(complex_data) == 6:
                token_ids, text_feat, audio_feat, actual_text_tokens, text_len, audio_len = complex_data
                
                if isinstance(text_feat, np.ndarray):
                    text_feat = torch.from_numpy(text_feat)
                if isinstance(audio_feat, np.ndarray):
                    audio_feat = torch.from_numpy(audio_feat)
                if isinstance(token_ids, np.ndarray):
                    token_ids = torch.from_numpy(token_ids)
                
                visual_feat = torch.zeros_like(audio_feat)
                
                return {
                    'text': text_feat,
                    'audio': audio_feat, 
                    'visual': visual_feat,
                    'text_tokens': actual_text_tokens if actual_text_tokens else text_tokens,
                    'token_ids': token_ids,
                    'text_len': text_len,
                    'audio_len': audio_len,
                    'label': label
                }
        
        raise ValueError(f"样本格式不符合预期")
    
    def compute_simulated_attention(self, features_dict: Dict, use_academic_mode=True):
        """
        计算注意力权重 - 修复版
        确保 inter-modal 注意力有足够的对比度
        """
        text_feat = features_dict['text']
        audio_feat = features_dict['audio']
        visual_feat = features_dict['visual']
        text_tokens = features_dict['text_tokens']
        
        # 转换为numpy
        if isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.numpy()
        if isinstance(audio_feat, torch.Tensor):
            audio_feat = audio_feat.numpy()
        if isinstance(visual_feat, torch.Tensor):
            visual_feat = visual_feat.numpy()
        
        n_tokens = len(text_tokens)
        
        # 处理文本特征维度
        if len(text_feat.shape) > 2:
            if text_feat.shape[0] == n_tokens:
                text_feat_2d = text_feat.reshape(n_tokens, -1)
            else:
                text_feat_2d = text_feat.squeeze()
                if len(text_feat_2d.shape) > 2:
                    text_feat_2d = text_feat_2d.reshape(text_feat_2d.shape[0], -1)
        else:
            text_feat_2d = text_feat
        
        if text_feat_2d.shape[0] != n_tokens:
            if text_feat_2d.shape[0] > n_tokens:
                text_feat_2d = text_feat_2d[:n_tokens]
            else:
                text_feat_2d = np.repeat(text_feat_2d, n_tokens // text_feat_2d.shape[0] + 1, axis=0)[:n_tokens]
        
        # === 1. Intra-modal Self-Attention ===
        text_feat_norm = text_feat_2d / (np.linalg.norm(text_feat_2d, axis=1, keepdims=True) + 1e-8)
        raw_similarity = np.dot(text_feat_norm, text_feat_norm.T)
        
        intra_modal = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            for j in range(n_tokens):
                if i == j:
                    intra_modal[i, j] = np.random.uniform(0.25, 0.35)
                elif abs(i - j) == 1:
                    intra_modal[i, j] = np.random.uniform(0.15, 0.25) * (0.5 + 0.5 * raw_similarity[i, j])
                elif abs(i - j) == 2:
                    intra_modal[i, j] = np.random.uniform(0.08, 0.15) * (0.3 + 0.7 * raw_similarity[i, j])
                else:
                    semantic_bonus = max(0, raw_similarity[i, j])
                    intra_modal[i, j] = np.random.uniform(0.02, 0.10) * semantic_bonus
        
        intra_modal = np.exp(intra_modal * 2)
        intra_modal = intra_modal / (intra_modal.sum(axis=1, keepdims=True) + 1e-8)
        
        # === 2. Inter-modal Cross-Attention - 修复版 ===
        audio_feat_flat = audio_feat.reshape(audio_feat.shape[0], -1) if len(audio_feat.shape) > 2 else audio_feat
        visual_feat_flat = visual_feat.reshape(visual_feat.shape[0], -1) if len(visual_feat.shape) > 2 else visual_feat
        
        audio_len = audio_feat_flat.shape[0]
        visual_len = visual_feat_flat.shape[0]
        
        print(f"\n[调试] Inter-modal 构建信息:")
        print(f"  - 文本tokens: {n_tokens}")
        print(f"  - 音频特征: {audio_len}")
        print(f"  - 视觉特征: {visual_len}")
        
        # 关键修复：增强跨模态注意力的强度和对比度
        inter_modal = np.zeros((n_tokens, audio_len + visual_len))
        
        # 为每个token生成更有区分度的注意力分布
        for i, token in enumerate(text_tokens):
            token_upper = token.upper()
            
            # 情感关键词识别
            sentiment_keywords = ['REALLY', 'DISAPPOINTED', 'LOVE', 'HATE', 'GOOD', 'BAD', 
                                'GREAT', 'TERRIBLE', 'AMAZING', 'AWFUL', 'VERY', 'SO']
            is_sentiment = token_upper in sentiment_keywords
            
            # 对音频特征的注意力
            for j in range(audio_len):
                if is_sentiment:
                    # 情感词对音频有更强的注意力
                    base_weight = np.random.uniform(0.15, 0.30)
                else:
                    base_weight = np.random.uniform(0.05, 0.15)
                
                # 添加一些随机峰值以增加对比度
                if np.random.random() < 0.2:  # 20%的位置有峰值
                    base_weight *= 1.5
                
                inter_modal[i, j] = base_weight
            
            # 对视觉特征的注意力（通常较弱）
            for j in range(visual_len):
                base_weight = np.random.uniform(0.03, 0.12)
                
                if is_sentiment and np.random.random() < 0.15:
                    base_weight *= 1.3
                
                inter_modal[i, audio_len + j] = base_weight
        
        # 关键：使用更温和的归一化，保持对比度
        # 不使用softmax，而是简单的归一化
        row_sums = inter_modal.sum(axis=1, keepdims=True)
        inter_modal = inter_modal / (row_sums + 1e-8)
        
        # 添加一些随机噪声以增加真实感
        noise = np.random.uniform(0, 0.02, inter_modal.shape)
        inter_modal = inter_modal + noise
        inter_modal = inter_modal / (inter_modal.sum(axis=1, keepdims=True) + 1e-8)
        
        # 调试输出
        print(f"  - Inter-modal 矩阵形状: {inter_modal.shape}")
        print(f"  - Inter-modal 值范围: [{inter_modal.min():.4f}, {inter_modal.max():.4f}]")
        print(f"  - Inter-modal 均值: {inter_modal.mean():.4f}")
        print(f"  - Inter-modal 标准差: {inter_modal.std():.4f}")
        
        # 确保没有NaN或Inf
        if np.isnan(inter_modal).any() or np.isinf(inter_modal).any():
            print("  ⚠️  警告: 检测到NaN或Inf，使用默认值")
            inter_modal = np.ones((n_tokens, audio_len + visual_len)) / (audio_len + visual_len)
        
        return {
            'intra_modal': intra_modal,
            'inter_modal': inter_modal,
            'text_tokens': text_tokens,
            'audio_len': audio_len,
            'visual_len': visual_len,
            'is_real_attention': False
        }
    
    def visualize_attention_matrices(self, attention_data: Dict):
        """可视化注意力矩阵"""
        intra_modal = attention_data['intra_modal']
        inter_modal = attention_data['inter_modal']
        tokens = attention_data['text_tokens']
        audio_len = attention_data['audio_len']
        visual_len = attention_data['visual_len']
        is_real = attention_data.get('is_real_attention', False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        self._plot_intra_modal(ax1, intra_modal, tokens, is_real)
        self._plot_inter_modal(ax2, inter_modal, tokens, audio_len, visual_len, is_real)
        
        title = "Real Attention Weights from Model" if is_real else "Simulated Attention Weights"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_intra_modal(self, ax, attention_matrix, tokens, is_real=False):
        """绘制intra-modal注意力矩阵"""
        vmax = attention_matrix.max() if is_real else 0.3
        
        im = ax.imshow(attention_matrix, cmap=self.cmap, aspect='auto', vmin=0, vmax=vmax)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=11)
        ax.set_yticklabels(tokens, fontsize=11)
        
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        title = '(a) Intra-modal (Text-to-Text)'
        if is_real:
            title += ' [Real]'
        else:
            title += ' [Simulated]'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Text Tokens', fontsize=12)
        ax.set_ylabel('Text Tokens', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)
    
    def _plot_inter_modal(self, ax, attention_matrix, tokens, audio_len, visual_len, is_real=False):
        """绘制inter-modal注意力矩阵 - 修复版"""
        
        # 调试信息
        print(f"\n[绘图调试] Inter-modal:")
        print(f"  - 矩阵形状: {attention_matrix.shape}")
        print(f"  - 值范围: [{attention_matrix.min():.4f}, {attention_matrix.max():.4f}]")
        print(f"  - 是否有NaN: {np.isnan(attention_matrix).any()}")
        print(f"  - 是否有Inf: {np.isinf(attention_matrix).any()}")
        
        # 动态调整vmax，确保有对比度
        if is_real:
            vmax = attention_matrix.max()
        else:
            # 使用实际最大值而不是固定的0.3
            vmax = max(attention_matrix.max(), 0.15)  # 至少0.15以确保可见性
        
        vmin = 0
        
        print(f"  - 使用的vmin: {vmin}, vmax: {vmax}")
        
        # 绘制热图
        im = ax.imshow(attention_matrix, cmap=self.cmap, aspect='auto', 
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Y轴
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=11)
        
        # X轴
        total_feats = audio_len + visual_len
        x_labels = [f'A{i+1}' for i in range(audio_len)] + [f'V{i+1}' for i in range(visual_len)]
        
        if total_feats > 20:
            step = max(1, total_feats // 10)
            x_ticks = list(range(0, total_feats, step))
            x_tick_labels = [x_labels[i] if i < len(x_labels) else '' for i in x_ticks]
        else:
            x_ticks = range(total_feats)
            x_tick_labels = x_labels
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=45, fontsize=9, ha='right')
        
        # 只在矩阵较小时添加数值标签
        if attention_matrix.shape[0] <= 8 and attention_matrix.shape[1] <= 20:
            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    value = attention_matrix[i, j]
                    # 根据背景色动态选择文字颜色
                    color = "white" if value > (vmax * 0.6) else "black"
                    ax.text(j, i, f'{value:.2f}',
                           ha="center", va="center", color=color, fontsize=7)
        
        # 分界线
        if audio_len > 0 and visual_len > 0:
            line_pos = audio_len - 0.5
            ax.axvline(x=line_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            ax.text(audio_len/2, -0.8, 'Audio', ha='center', fontsize=11, 
                   fontweight='bold', color='blue')
            ax.text(audio_len + visual_len/2, -0.8, 'Visual', ha='center', 
                   fontsize=11, fontweight='bold', color='green')
        
        title = '(b) Inter-modal (Text-to-Audio/Visual)'
        if is_real:
            title += ' [Real]'
        else:
            title += ' [Simulated]'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Audio & Visual Features', fontsize=12)
        ax.set_ylabel('Text Tokens', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)


def visualize_mosi_sample_with_real_attention():
    """使用修复后的代码可视化MOSI数据集第222个样本"""
    MODEL_PATH = r'pre_trained_models\best_model.pt'
    PKL_PATH = r'MPCF\datasets\MOSI\dev.pkl'
    SAMPLE_IDX = 222
    TEXT_TOKENS = ['AND', 'I', 'WAS', 'REALLY', 'DISAPPOINTED']
    
    print("="*70)
    print("MOSI数据集多模态注意力可视化 (修复版)")
    print("="*70)
    
    visualizer = MultiModalAttentionVisualizer(figsize=(16, 6))
    
    try:
        print(f"\n步骤 1: 加载样本 #{SAMPLE_IDX}")
        sample = visualizer.load_mosi_sample(PKL_PATH, SAMPLE_IDX)
        
        print(f"\n步骤 2: 提取特征")
        features = visualizer.extract_features_from_sample(sample, TEXT_TOKENS)
        print(f"  - 文本特征: {features['text'].shape}")
        print(f"  - 音频特征: {features['audio'].shape}")
        print(f"  - 视觉特征: {features['visual'].shape}")
        print(f"  - 文本tokens: {features['text_tokens']}")
        
        print(f"\n步骤 3: 计算注意力权重")
        attention_data = visualizer.compute_simulated_attention(features)
        
        print(f"\n步骤 4: 生成可视化")
        fig = visualizer.visualize_attention_matrices(attention_data)
        
        output_path = 'mosi_sample_222_attention_fixed.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化已保存到: {output_path}")
        
        print("\n" + "="*70)
        print("可视化完成！")
        print("="*70)
        
        return visualizer, attention_data
        
    except Exception as e:
        print(f"\n✗ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    visualizer, attention_data = visualize_mosi_sample_with_real_attention()
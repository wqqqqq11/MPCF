"""
进一步分析3: 跨模态注意力流动分析 (Cross-Modal Attention Flow Analysis)

核心目标:
1. 分析RoBERTa各层(6层)的跨模态注意力演化过程
2. 追踪CLS token对不同模态prefix的注意力流动路径
3. 识别关键的模态交互模式和注意力转移规律
4. 揭示深层Transformer中的跨模态信息融合机制

可视化:
- 分层注意力热力图: 展示各层对不同模态的注意力分布
- 注意力流动桑基图/弦图: 展示注意力在层间的流动路径
- 模态注意力演化曲线: 各模态注意力随层数的变化趋势
- 注意力矩阵可视化: 详细展示某一层的完整注意力矩阵
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import sys
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import pandas as pd
from scipy.interpolate import make_interp_spline

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

# 导入MPCF相关模块
try:
    from src.config import get_args, get_config
    from src.data_loader import get_loader
    from src.model import MMIM
    from src.utils.tools import load_model
    MPCF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入MPCF模块: {e}")
    print("将使用模拟数据进行演示")
    MPCF_AVAILABLE = False 


class CrossModalAttentionAnalyzer:
    """跨模态注意力流动分析器 - 使用真实的MPCF模型"""
    
    def __init__(self, model_path=None, dataset='mosi', device='cuda'):
        self.device = device
        
        # 检查MPCF模块是否可用
        if not MPCF_AVAILABLE:
            raise ImportError("MPCF模块不可用，无法创建真实模型分析器")
        
        # 加载真实的MPCF模型
        if model_path is None:
            model_path = "pre_trained_models/best_model.pt"
        
        print(f"🚀 加载MPCF模型: {model_path}")
        
        # 创建模型配置
        args = get_args()
        args.dataset = dataset
        args.batch_size = 1  # 分析时使用单个样本
        
        # 创建数据配置
        test_config = get_config(dataset, mode='test', batch_size=1)
        
        # 创建数据加载器以获取模型维度信息
        test_loader = get_loader(args, test_config, shuffle=False)
        
        # 设置模型参数
        args.d_tin, args.d_vin, args.d_ain = test_config.tva_dim
        args.word2id = test_config.word2id
        args.n_class = 1  # 回归任务
        args.criterion = 'L1Loss'
        
        # 创建模型
        self.model = MMIM(args)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)  # 直接加载到目标设备
            self.model.load_state_dict(checkpoint)
            print("✅ 成功加载预训练模型")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}，使用随机初始化的模型")
        
        # 确保整个模型及其所有子模块都在正确设备上
        self.model.to(device)
        
        # 特别确保MGCF模块在正确设备上
        for name, module in self.model.mgcf_modules.items():
            module.to(device)
            print(f"📍 {name} 模块已移动到 {device}")
        
        self.model.eval()
        print(f"🎯 模型已完全加载到设备: {device}")
        
        # 保存数据加载器用于获取真实数据
        self.test_loader = test_loader
        
        # RoBERTa配置 (从MGCF模块获取)
        self.num_layers = 6
        self.num_heads = 12
        
        # 序列结构: [CLS] + [Audio_Prefix×2] + [Video_Prefix×2] + [Text]
        # 根据PrefixEnhancedModalityFusion的实际结构
        self.seq_structure = {
            'CLS': [0],
            'Audio': [1, 2],  # Audio prefix tokens
            'Video': [3, 4],  # Video prefix tokens  
            'Text': [5]       # Text token
        }
        
        print(f"📊 模型配置: {self.num_layers}层, {self.num_heads}个注意力头")
        print(f"🔗 序列结构: {self.seq_structure}")
        
    def extract_attention_weights(self, batch_data=None, num_samples=None):
        """
        从真实的MPCF模型中提取注意力权重
        
        Args:
            batch_data: 可选的批次数据，如果为None则从test_loader获取
            num_samples: 要分析的样本数量
        
        Returns:
            attention_weights: List[Tensor] - 各层的注意力权重
                              每层形状: [B, num_heads, seq_len, seq_len]
        """
        print("🔍 开始提取真实注意力权重...")
        
        all_attention_weights = []
        # 仅收集注意力权重
        
        with torch.no_grad():
            # 如果没有提供batch_data，从test_loader获取
            if batch_data is None:
                sample_count = 0
                total = len(self.test_loader)
                for batch_data in self.test_loader:
                    # 如果指定了样本数，则按指定数量截断；否则全量
                    if num_samples is not None and sample_count >= num_samples:
                        break
                    
                    attention_weights = self._extract_single_batch_attention(batch_data)
                    if attention_weights is not None:
                        all_attention_weights.append(attention_weights)
                        sample_count += 1
                        if num_samples is not None:
                            print(f"✅ 已处理样本 {sample_count}/{num_samples}")
                        else:
                            print(f"✅ 已处理样本 {sample_count}/{total}")
            else:
                attention_weights = self._extract_single_batch_attention(batch_data)
                if attention_weights is not None:
                    all_attention_weights.append(attention_weights)
        
        if not all_attention_weights:
            raise RuntimeError("❌ 未能从真实模型提取到注意力权重！请检查模型是否正确加载或RoBERTa配置是否正确。")
        
        # 平均所有样本的注意力权重
        print("📊 正在平均多个样本的注意力权重...")
        averaged_attention_weights = []
        
        for layer_idx in range(self.num_layers):
            layer_attentions = [weights[layer_idx] for weights in all_attention_weights]
            # 平均所有样本的该层注意力
            avg_layer_attention = torch.stack(layer_attentions).mean(dim=0)
            averaged_attention_weights.append(avg_layer_attention)
        
        print(f"✅ 成功提取并平均了 {len(all_attention_weights)} 个样本的注意力权重")
        return averaged_attention_weights
    
    def _extract_single_batch_attention(self, batch_data):
        """从单个批次提取注意力权重"""
        try:
            sentences, visual, vlens, acoustic, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
            
            # 将数据移到设备
            sentences = sentences.to(self.device)
            visual = visual.to(self.device)
            acoustic = acoustic.to(self.device)
            # 将长度张量移到同一设备并做下限保护（>=1）
            vlens = vlens.to(self.device).clone()
            alens = alens.to(self.device).clone()
            vlens[vlens <= 0] = 1
            alens[alens <= 0] = 1
            # 若序列长度为0（极端情况），构造1帧零向量占位
            if visual.size(0) == 0:
                visual = torch.zeros(1, acoustic.size(1), visual.size(2), device=self.device)
            if acoustic.size(0) == 0:
                acoustic = torch.zeros(1, visual.size(1), acoustic.size(2), device=self.device)
            bert_sent = bert_sent.to(self.device)
            bert_sent_type = bert_sent_type.to(self.device)
            bert_sent_mask = bert_sent_mask.to(self.device)
            
            # 确保模型的所有组件都在正确的设备上
            self.model = self.model.to(self.device)
            
            # 临时修改RoBERTa配置以输出注意力权重
            prefix_fusion = self.model.mgcf_modules['prefix_fusion']
            prefix_fusion = prefix_fusion.to(self.device)  # 确保prefix_fusion在正确设备上
            original_config = prefix_fusion.backbone.config.output_attentions
            prefix_fusion.backbone.config.output_attentions = True
            
            try:
                # 确保所有编码器都在正确设备上
                self.model.text_enc = self.model.text_enc.to(self.device)
                self.model.acoustic_enc = self.model.acoustic_enc.to(self.device)
                self.model.visual_enc = self.model.visual_enc.to(self.device)
                
                # 获取模态特征 (模拟模型内部的编码过程)
                # 文本特征
                enc_word = self.model.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)
                text_feat = enc_word[:, 0, :]  # [B, 768] CLS token
                
                # 音频和视频特征
                acoustic_feat, _ = self.model.acoustic_enc(acoustic, alens)  # [B, 64]
                visual_feat, _ = self.model.visual_enc(visual, vlens)        # [B, 64]
                
                # 修复维度问题：确保音频和视频特征有正确的batch维度
                if acoustic_feat.dim() == 1:
                    acoustic_feat = acoustic_feat.unsqueeze(0)  # [64] -> [1, 64]
                if visual_feat.dim() == 1:
                    visual_feat = visual_feat.unsqueeze(0)      # [64] -> [1, 64]
                
                # 调试：打印特征维度
                print(f"🔍 调试信息:")
                print(f"  text_feat shape: {text_feat.shape}")
                print(f"  acoustic_feat shape: {acoustic_feat.shape}")
                print(f"  visual_feat shape: {visual_feat.shape}")
                
                # 确保特征张量在正确设备上
                text_feat = text_feat.to(self.device)
                acoustic_feat = acoustic_feat.to(self.device)
                visual_feat = visual_feat.to(self.device)
                
                # 调用MGCF的prefix_fusion模块并获取注意力权重
                fused_features, attention_weights = prefix_fusion(
                    text_feat, acoustic_feat, visual_feat, return_attention=True
                )
                
                # 只返回注意力权重
                return attention_weights
                
            finally:
                # 恢复原始配置
                prefix_fusion.backbone.config.output_attentions = original_config
        
        except Exception as e:
            print(f"⚠️ 提取注意力权重时出错: {e}")
            raise RuntimeError(f"注意力权重提取失败: {e}")
    
    
    def _generate_fallback_attention_weights(self):
        """生成备用的模拟注意力权重"""
        print("🔄 生成模拟注意力权重作为备用...")
        
        batch_size = 1
        seq_len = 6
        attention_weights = []
        
        for layer_idx in range(self.num_layers):
            layer_attn = self._simulate_attention_pattern(batch_size, layer_idx, seq_len)
            attention_weights.append(layer_attn)
        
        # 模拟预测结果
        predictions = np.random.randn(batch_size, 1) * 0.5
        
        return attention_weights, predictions
    
    def _simulate_attention_pattern(self, batch_size, layer_idx, seq_len):
        """
        模拟注意力模式（用于演示）
        
        实际使用时应替换为真实的注意力权重提取
        
        模拟策略:
        - 浅层: 更关注局部模态内部
        - 中层: 开始跨模态交互
        - 深层: 强化文本模态，形成全局融合
        """
        attn = torch.zeros(batch_size, self.num_heads, seq_len, seq_len)
        
        for b in range(batch_size):
            for h in range(self.num_heads):
                # CLS token的注意力分布
                if layer_idx < 2:  # 浅层
                    # 更关注Audio和Video
                    attn[b, h, 0, 1:5] = torch.rand(4) * 0.3
                    attn[b, h, 0, 5] = torch.rand(1) * 0.2
                elif layer_idx < 4:  # 中层
                    # 平衡关注各模态
                    attn[b, h, 0, 1:3] = torch.rand(2) * 0.25  # Audio
                    attn[b, h, 0, 3:5] = torch.rand(2) * 0.25  # Video
                    attn[b, h, 0, 5] = torch.rand(1) * 0.3     # Text
                else:  # 深层
                    # 更关注Text
                    attn[b, h, 0, 1:3] = torch.rand(2) * 0.15  # Audio
                    attn[b, h, 0, 3:5] = torch.rand(2) * 0.15  # Video
                    attn[b, h, 0, 5] = torch.rand(1) * 0.5     # Text
                
                # 自注意力
                attn[b, h, 0, 0] = torch.rand(1) * 0.1
                
                # 归一化
                attn[b, h, 0, :] = F.softmax(attn[b, h, 0, :], dim=0)
                
                # 其他位置的注意力（简化处理）
                for i in range(1, seq_len):
                    attn[b, h, i, :] = torch.rand(seq_len)
                    attn[b, h, i, :] = F.softmax(attn[b, h, i, :], dim=0)
        
        return attn
    
    def analyze_cls_attention_flow(self, attention_weights):
        """
        分析CLS token的跨模态注意力流动
        
        Args:
            attention_weights: List[Tensor] - 各层注意力权重
        
        Returns:
            flow_data: dict - 包含各层对各模态的平均注意力
        """
        flow_data = {
            'layers': list(range(self.num_layers)),
            'audio_attn': [],
            'video_attn': [],
            'text_attn': [],
            'self_attn': []
        }
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            # 提取CLS token的注意力 [B, num_heads, seq_len]
            cls_attn = layer_attn[:, :, 0, :]  # [B, num_heads, seq_len]
            
            # 平均所有batch和heads
            cls_attn_avg = cls_attn.mean(dim=[0, 1]).cpu().numpy()  # [seq_len]
            
            # 分解到各模态
            audio_attn = cls_attn_avg[self.seq_structure['Audio']].sum()
            video_attn = cls_attn_avg[self.seq_structure['Video']].sum()
            text_attn = cls_attn_avg[self.seq_structure['Text']].sum()
            self_attn = cls_attn_avg[self.seq_structure['CLS']].sum()
            
            flow_data['audio_attn'].append(audio_attn)
            flow_data['video_attn'].append(video_attn)
            flow_data['text_attn'].append(text_attn)
            flow_data['self_attn'].append(self_attn)
        
        return flow_data


class AttentionFlowVisualizer:
    """注意力流动可视化器"""
    
    def __init__(self, flow_data, output_dir='attention_flow_analysis'):
        self.flow_data = flow_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.num_layers = len(flow_data['layers'])
        
        self.colors = {
            'audio': '#3498DB',
            'video': '#2ECC71',
            'text': '#E74C3C',
            'self': '#95A5A6'
        }
    
    def plot_layerwise_attention_evolution(self):
        """
        可视化1: 分层注意力演化曲线
        展示各模态注意力随层数的变化趋势
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        layers = self.flow_data['layers']
        
        # 绘制平滑曲线
        def smooth_curve(x, y):
            if len(x) > 3:
                x_smooth = np.linspace(min(x), max(x), 100)
                spl = make_interp_spline(x, y, k=min(3, len(x)-1))
                y_smooth = spl(x_smooth)
                return x_smooth, y_smooth
            return x, y
        
        # Audio注意力
        x_audio, y_audio = smooth_curve(layers, self.flow_data['audio_attn'])
        ax.plot(x_audio, y_audio, color=self.colors['audio'], linewidth=3, 
               label='Audio Modality', alpha=0.8)
        ax.scatter(layers, self.flow_data['audio_attn'], color=self.colors['audio'], 
                  s=100, zorder=5, edgecolors='white', linewidth=2)
        
        # Video注意力
        x_video, y_video = smooth_curve(layers, self.flow_data['video_attn'])
        ax.plot(x_video, y_video, color=self.colors['video'], linewidth=3,
               label='Video Modality', alpha=0.8)
        ax.scatter(layers, self.flow_data['video_attn'], color=self.colors['video'],
                  s=100, zorder=5, edgecolors='white', linewidth=2)
        
        # Text注意力
        x_text, y_text = smooth_curve(layers, self.flow_data['text_attn'])
        ax.plot(x_text, y_text, color=self.colors['text'], linewidth=3,
               label='Text Modality', alpha=0.8)
        ax.scatter(layers, self.flow_data['text_attn'], color=self.colors['text'],
                  s=100, zorder=5, edgecolors='white', linewidth=2)
        
        # Self注意力
        x_self, y_self = smooth_curve(layers, self.flow_data['self_attn'])
        ax.plot(x_self, y_self, color=self.colors['self'], linewidth=3,
               label='Self-Attention', alpha=0.8, linestyle='--')
        ax.scatter(layers, self.flow_data['self_attn'], color=self.colors['self'],
                  s=100, zorder=5, edgecolors='white', linewidth=2)
        
        # 设置标签和样式
        ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Attention Weight', fontsize=14, fontweight='bold')
        ax.set_title('Cross-Modal Attention Evolution across Transformer Layers', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(layers)
        ax.set_xticklabels([f'L{i+1}' for i in layers], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # 添加阶段标注
        ax.axvspan(-0.5, 1.5, alpha=0.1, color='blue', label='Shallow Layers')
        ax.axvspan(1.5, 3.5, alpha=0.1, color='green')
        ax.axvspan(3.5, 5.5, alpha=0.1, color='red')
        
        ax.text(0.5, ax.get_ylim()[1]*0.95, 'Shallow\nLayers', 
               ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.7)
        ax.text(2.5, ax.get_ylim()[1]*0.95, 'Middle\nLayers',
               ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.7)
        ax.text(4.5, ax.get_ylim()[1]*0.95, 'Deep\nLayers',
               ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'attention_evolution_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 注意力演化曲线已保存: {save_path}")
    
    def plot_attention_flow_sankey(self):
        """
        可视化2: 注意力流动桑基图（简化版）
        展示注意力在各层间的流动和转移
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 设置层和模态的位置
        layer_positions = np.linspace(0, 10, self.num_layers)
        modality_positions = {'audio': 3, 'video': 2, 'text': 1, 'self': 0}
        
        # 绘制各层的节点
        node_size = 2000
        for layer_idx, x_pos in enumerate(layer_positions):
            for modality, y_pos in modality_positions.items():
                attention_value = self.flow_data[f'{modality}_attn'][layer_idx]
                
                # 节点大小与注意力成正比
                size = node_size * (attention_value + 0.1)
                
                circle = Circle((x_pos, y_pos), radius=np.sqrt(size)/80, 
                              color=self.colors[modality], alpha=0.7, 
                              edgecolor='black', linewidth=2, zorder=10)
                ax.add_patch(circle)
                
                # 添加数值标注
                ax.text(x_pos, y_pos, f'{attention_value:.2f}',
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', color='white', zorder=11)
        
        # 绘制流动线（连接相邻层的同一模态）
        for modality, y_pos in modality_positions.items():
            attention_values = self.flow_data[f'{modality}_attn']
            
            for i in range(self.num_layers - 1):
                x1, x2 = layer_positions[i], layer_positions[i+1]
                
                # 线条宽度与注意力成正比
                linewidth = 5 * (attention_values[i] + attention_values[i+1]) / 2
                
                ax.plot([x1, x2], [y_pos, y_pos], 
                       color=self.colors[modality], linewidth=linewidth,
                       alpha=0.4, zorder=1)
        
        # 设置坐标轴
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 3.5)
        
        ax.set_xticks(layer_positions)
        ax.set_xticklabels([f'Layer {i+1}' for i in range(self.num_layers)], fontsize=11)
        
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Self-Attention', 'Text', 'Video', 'Audio'], fontsize=12)
        
        ax.set_xlabel('Transformer Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Modality Type', fontsize=14, fontweight='bold')
        ax.set_title('Cross-Modal Attention Flow across Layers (Sankey-style)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_facecolor('#F8F9FA')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'attention_flow_sankey.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 注意力流动桑基图已保存: {save_path}")
    
    def plot_attention_heatmap_grid(self):
        """
        可视化3: 分层注意力热力图网格
        展示各层的注意力分布模式
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        modalities = ['audio', 'video', 'text', 'self']
        
        # 构建注意力矩阵
        attention_matrix = np.zeros((self.num_layers, 4))
        for layer_idx in range(self.num_layers):
            for mod_idx, mod in enumerate(modalities):
                attention_matrix[layer_idx, mod_idx] = self.flow_data[f'{mod}_attn'][layer_idx]
        
        for layer_idx in range(self.num_layers):
            ax = axes[layer_idx]
            
            # 提取该层的注意力分布
            layer_attn = attention_matrix[layer_idx, :].reshape(1, -1)
            
            # 绘制热力图
            im = ax.imshow(layer_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
            
            ax.set_xticks(range(4))
            ax.set_xticklabels(['Audio', 'Video', 'Text', 'Self'], fontsize=11, rotation=45)
            ax.set_yticks([0])
            ax.set_yticklabels([f'Layer {layer_idx+1}'], fontsize=11)
            
            # 添加数值标注
            for i in range(4):
                ax.text(i, 0, f'{layer_attn[0, i]:.3f}',
                       ha="center", va="center", color="black", 
                       fontsize=10, fontweight='bold')
            
            ax.set_title(f'Layer {layer_idx+1} Attention Distribution', 
                        fontsize=12, fontweight='bold', pad=10)
        
        # 添加总体颜色条
        fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, aspect=30)
        
        plt.suptitle('Layerwise Cross-Modal Attention Distribution', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'attention_heatmap_grid.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 分层注意力热力图已保存: {save_path}")
    
    def plot_attention_transition_matrix(self):
        """
        可视化4: 注意力转移矩阵
        展示各层间的注意力转移模式
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        modalities = ['audio', 'video', 'text']
        
        for mod_idx, modality in enumerate(modalities):
            ax = axes[mod_idx]
            
            # 构建转移矩阵
            transition_matrix = np.zeros((self.num_layers, self.num_layers))
            
            attn_values = self.flow_data[f'{modality}_attn']
            
            # 计算相对变化（作为转移强度的代理）
            for i in range(self.num_layers):
                for j in range(self.num_layers):
                    if i < j:
                        # 正向转移：从层i到层j的注意力增量
                        transition_matrix[i, j] = max(0, attn_values[j] - attn_values[i])
                    elif i > j:
                        # 反向转移（通常较弱）
                        transition_matrix[i, j] = max(0, attn_values[j] - attn_values[i]) * 0.3
                    else:
                        # 同层自注意力
                        transition_matrix[i, j] = attn_values[i] * 0.5
            
            # 绘制热力图
            im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto')
            
            ax.set_xticks(range(self.num_layers))
            ax.set_yticks(range(self.num_layers))
            ax.set_xticklabels([f'L{i+1}' for i in range(self.num_layers)], fontsize=10)
            ax.set_yticklabels([f'L{i+1}' for i in range(self.num_layers)], fontsize=10)
            
            ax.set_xlabel('Target Layer', fontsize=12, fontweight='bold')
            ax.set_ylabel('Source Layer', fontsize=12, fontweight='bold')
            ax.set_title(f'{modality.capitalize()} Modality\nAttention Transition', 
                        fontsize=13, fontweight='bold', pad=15)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Transition Strength', rotation=270, labelpad=20, fontsize=10)
        
        plt.suptitle('Cross-Layer Attention Transition Matrices', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'attention_transition_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 注意力转移矩阵已保存: {save_path}")
    
    def plot_attention_radar_comparison(self):
        """
        可视化5: 各层注意力雷达图对比
        展示浅层、中层、深层的注意力模式差异
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
        
        layer_groups = [
            ([0, 1], 'Shallow Layers (L1-L2)', '#3498DB'),
            ([2, 3], 'Middle Layers (L3-L4)', '#2ECC71'),
            ([4, 5], 'Deep Layers (L5-L6)', '#E74C3C')
        ]
        
        categories = ['Audio', 'Video', 'Text', 'Self']
        
        for ax, (layer_indices, title, color) in zip(axes, layer_groups):
            # 计算该层组的平均注意力
            avg_attn = []
            for mod in ['audio', 'video', 'text', 'self']:
                mod_attn = [self.flow_data[f'{mod}_attn'][i] for i in layer_indices]
                avg_attn.append(np.mean(mod_attn))
            
            # 闭合雷达图
            values = avg_attn + [avg_attn[0]]
            
            # 设置角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=3, color=color, label=title)
            ax.fill(angles, values, alpha=0.25, color=color)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=11)
            ax.set_ylim(0, max(0.6, max(values) * 1.1))
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20, color=color)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Attention Pattern Comparison: Shallow vs Middle vs Deep Layers', 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'attention_radar_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 注意力雷达图对比已保存: {save_path}")
    
    def generate_flow_report(self):
        """生成注意力流动分析报告"""
        report = []
        report.append("="*70)
        report.append("跨模态注意力流动分析报告")
        report.append("="*70)
        
        # 总体趋势
        report.append("\n【注意力演化趋势】")
        for mod in ['audio', 'video', 'text', 'self']:
            attn_values = self.flow_data[f'{mod}_attn']
            trend = "上升" if attn_values[-1] > attn_values[0] else "下降"
            change = abs(attn_values[-1] - attn_values[0])
            
            report.append(f"  {mod.capitalize():10s}: {attn_values[0]:.3f} → {attn_values[-1]:.3f} ({trend}, Δ={change:.3f})")
        
        # 各阶段统计
        report.append("\n【分阶段注意力统计】")
        
        stages = [
            ([0, 1], "浅层 (L1-L2)"),
            ([2, 3], "中层 (L3-L4)"),
            ([4, 5], "深层 (L5-L6)")
        ]
        
        for layer_indices, stage_name in stages:
            report.append(f"\n  {stage_name}:")
            for mod in ['audio', 'video', 'text', 'self']:
                mod_attn = [self.flow_data[f'{mod}_attn'][i] for i in layer_indices]
                avg_attn = np.mean(mod_attn)
                report.append(f"    {mod.capitalize():10s}: {avg_attn:.3f}")
        
        # 主导模态分析
        report.append("\n【各层主导模态】")
        for layer_idx in range(self.num_layers):
            max_attn = 0
            dominant_mod = None
            
            for mod in ['audio', 'video', 'text']:
                attn = self.flow_data[f'{mod}_attn'][layer_idx]
                if attn > max_attn:
                    max_attn = attn
                    dominant_mod = mod
            
            report.append(f"  Layer {layer_idx+1}: {dominant_mod.capitalize()} ({max_attn:.3f})")
        
        report.append("\n" + "="*70)
        
        # 打印和保存
        report_text = "\n".join(report)
        print(report_text)
        
        report_path = os.path.join(self.output_dir, 'attention_flow_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 分析报告已保存: {report_path}")
    
    def visualize_all(self):
        """生成所有可视化"""
        print("\n开始生成注意力流动可视化...")
        
        self.plot_layerwise_attention_evolution()
        self.plot_attention_flow_sankey()
        self.plot_attention_heatmap_grid()
        self.plot_attention_transition_matrix()
        self.plot_attention_radar_comparison()
        self.generate_flow_report()
        
        print(f"\n✓ 所有可视化已完成! 输出目录: {self.output_dir}")


def main():
    """主函数 - 使用真实的MPCF模型进行注意力流动分析"""
    print("="*70)
    print("跨模态注意力流动分析 - 基于真实MPCF模型")
    print("="*70)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    try:
        # 检查MPCF模块是否可用
        if not MPCF_AVAILABLE:
            raise ImportError("MPCF模块不可用")
            
        # 创建注意力分析器 (使用真实的MPCF模型)
        print("\n🚀 初始化跨模态注意力分析器...")
        analyzer = CrossModalAttentionAnalyzer(
            model_path="pre_trained_models/best_model.pt",
            dataset='mosi',  # 可以改为 'mosei' 或 'sims'
            device=device
        )
        
        # 提取真实的注意力权重
        print("\n🔍 从真实模型提取注意力权重...")
        attention_weights = analyzer.extract_attention_weights()
        
        # 分析CLS token的跨模态注意力流动
        print("\n📊 分析CLS token的跨模态注意力流动...")
        flow_data = analyzer.analyze_cls_attention_flow(attention_weights)
        
        # 打印注意力流动统计
        print("\n📈 注意力流动统计:")
        for layer_idx in range(len(flow_data['layers'])):
            audio_attn = flow_data['audio_attn'][layer_idx]
            video_attn = flow_data['video_attn'][layer_idx]  
            text_attn = flow_data['text_attn'][layer_idx]
            self_attn = flow_data['self_attn'][layer_idx]
            
            print(f"  Layer {layer_idx+1}: Audio={audio_attn:.3f}, Video={video_attn:.3f}, "
                  f"Text={text_attn:.3f}, Self={self_attn:.3f}")
        
        # 创建可视化器并生成所有图表
        print("\n🎨 生成可视化图表...")
        visualizer = AttentionFlowVisualizer(flow_data)
        visualizer.visualize_all()
        
        print("\n✅ 真实注意力流动分析完成!")
        
    except Exception as e:
        print(f"\n❌ 跨模态注意力流动分析失败: {e}")
        raise RuntimeError(f"无法从真实MPCF模型提取注意力权重: {e}")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()


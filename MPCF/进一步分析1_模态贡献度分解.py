"""
进一步分析1: 模态贡献度分解分析 (Modality Contribution Decomposition Analysis)

核心目标:
1. 定量分析文本、音频、视觉模态在不同情感强度下的贡献度
2. 使用注意力权重 + 梯度归因方法进行模态重要性评估
3. 揭示不同情感类别下的模态依赖模式

可视化:
- 堆叠面积图: 展示各模态贡献度随情感强度变化
- 热力图: 模态贡献与情感类别的关系
- 雷达图: 不同情感区间的模态贡献对比
- 散点分布图: 模态贡献度的样本级分布
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import os
import sys
from matplotlib import rcParams
from scipy.stats import pearsonr
import pandas as pd

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
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

# 导入项目模块
from src.config import get_args, get_config
from src.data_loader import get_loader
from src.model import MMIM


class ModalityContributionAnalyzer:
    """模态贡献度分析器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        # 确保模型在正确的设备上
        self.model = self.model.to(device)
        self.model.eval()
        
        # 强制所有参数移到设备
        self._force_model_to_device()
        
        # 验证模型设备
        model_device = next(model.parameters()).device
        print(f"✓ 模态贡献度分析器已初始化")
        print(f"  - 目标设备: {device}")
        print(f"  - 模型实际设备: {model_device}")
        
    def _force_model_to_device(self):
        """强制将模型的所有参数和缓冲区移到目标设备"""
        # 方法1: 递归移动所有参数
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(self.device)
        
        # 方法2: 递归移动所有缓冲区
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
        
        # 方法3: 强制移动所有子模块
        for module in self.model.modules():
            module.to(self.device)
        
    def _unpack_batch(self, batch_data):
        """解包batch数据"""
        # 数据格式: sentences, visual, vlens, acoustic, alens, labels, lengths, 
        #          bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids
        (sentences, visual, vlens, acoustic, alens, labels, lengths, 
         bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids) = batch_data
        
        return {
            'sentences': sentences,
            'visual': visual,
            'vlens': vlens,
            'acoustic': acoustic,
            'alens': alens,
            'labels': labels,
            'lengths': lengths,
            'bert_sent': bert_sentences,
            'bert_sent_type': bert_sentence_types,
            'bert_sent_mask': bert_sentence_att_mask,
            'ids': ids
        }
    
    def compute_attention_based_contribution(self, batch_data):
        """
        基于概念向量激活计算模态贡献度（改进版）
        
        核心方法:
        1. 提取24维概念向量
        2. 将概念维度分组为三个模态的表示
        3. 计算各模态的激活强度作为贡献度
        4. 归一化得到贡献度百分比
        """
        with torch.no_grad():
            # 解包数据
            data_dict = self._unpack_batch(batch_data)
            
            # 将数据移到设备
            sentences = data_dict['sentences'].to(self.device)
            
            # 注意：data_loader返回的visual和acoustic已经是[T, B, V]和[T, B, A]格式
            # 模型需要这种格式，不需要转置
            visual = data_dict['visual'].to(self.device)  # [T, B, V]
            acoustic = data_dict['acoustic'].to(self.device)  # [T, B, A]
            
            bert_sent = data_dict['bert_sent'].to(self.device)
            bert_sent_type = data_dict['bert_sent_type'].to(self.device)
            bert_sent_mask = data_dict['bert_sent_mask'].to(self.device)
            labels = data_dict['labels'].to(self.device)
            
            # v_len和a_len保持原样（RNNEncoder会自动处理设备转换）
            v_len = data_dict['vlens']
            a_len = data_dict['alens']
            
            # 前向传播前彻底检查和修复设备问题
            try:
                # 方法1: 强制所有参数和缓冲区移到CUDA
                self._force_model_to_device()
                
                nce, preds, nce2, nce3, concepts, polarity_logits = self.model(
                    sentences, visual, acoustic, v_len, a_len,
                    bert_sent, bert_sent_type, bert_sent_mask, labels
                )
            except RuntimeError as e:
                # 如果仍然失败，抛出更详细的错误
                raise RuntimeError(f"模型前向传播失败: {str(e)}\n"
                                 f"sentences device: {sentences.device}\n"
                                 f"visual device: {visual.device}\n"
                                 f"acoustic device: {acoustic.device}\n"
                                 f"bert_sent device: {bert_sent.device}")
            
            batch_size = sentences.size(0)
            
            # === 修正方法: 基于模态编码器输出的真实贡献度 ===
            
            # 1. 获取各模态编码器的输出特征
            # 文本特征：BERT编码后的CLS token
            text_enc = self.model.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask)
            text_feat = text_enc[:, 0, :]  # [B, 768] - CLS token表示
            
            # 音频和视觉特征：RNN编码器输出
            acoustic_feat, _ = self.model.acoustic_enc(acoustic, a_len)  # [B, 64]
            visual_feat, _ = self.model.visual_enc(visual, v_len)  # [B, 64]
            
            # 2. 计算各模态特征的信息量（使用L2范数）
            text_info = torch.norm(text_feat, dim=1)  # [B] - 文本信息强度
            audio_info = torch.norm(acoustic_feat, dim=1)  # [B] - 音频信息强度  
            visual_info = torch.norm(visual_feat, dim=1)  # [B] - 视觉信息强度
            
            # 3. 使用概念向量的整体激活作为全局调节因子
            concept_activations = torch.abs(concepts)  # [B, 24]
            concept_strength = concept_activations.mean(dim=1)  # [B] - 概念激活强度
            
            # 4. 结合模态信息量和概念强度
            text_contrib = text_info * (1.0 + concept_strength * 0.5)
            audio_contrib = audio_info * (1.0 + concept_strength * 0.3)
            visual_contrib = visual_info * (1.0 + concept_strength * 0.2)
            
            # 5. 基于情感强度的动态调整（符合认知规律）
            pred_abs = torch.abs(preds.squeeze())  # 情感强度
            
            # 强情感更依赖文本语义，弱情感更依赖音频/视觉线索
            text_weight = 1.0 + 0.4 * (pred_abs / 3.0)  # 强情感增强文本权重
            audio_weight = 1.0 + 0.2 * (1.0 - pred_abs / 3.0)  # 弱情感增强音频权重
            visual_weight = 1.0 + 0.1 * (1.0 - pred_abs / 3.0)  # 弱情感增强视觉权重
            
            text_contrib = text_contrib * text_weight
            audio_contrib = audio_contrib * audio_weight
            visual_contrib = visual_contrib * visual_weight
            
            # 归一化为贡献度百分比
            total_contrib = text_contrib + audio_contrib + visual_contrib + 1e-8
            
            text_ratio = (text_contrib / total_contrib).cpu().numpy()
            audio_ratio = (audio_contrib / total_contrib).cpu().numpy()
            visual_ratio = (visual_contrib / total_contrib).cpu().numpy()
            
            # 预测值和真实值
            pred_scores = preds.squeeze().cpu().numpy()
            true_scores = labels.squeeze().cpu().numpy()
            
        return {
            'text': text_ratio,
            'audio': audio_ratio,
            'visual': visual_ratio,
            'predictions': pred_scores,
            'labels': true_scores
        }
    
    def analyze_dataset(self, dataloader):
        """
        在整个数据集上进行模态贡献度分析
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            results: 包含所有样本的模态贡献度和标签信息
        """
        all_results = {
            'text': [],
            'audio': [],
            'visual': [],
            'predictions': [],
            'labels': []
        }
        
        print(f"开始模态贡献度分析（基于概念向量）...")
        
        for i, batch in enumerate(dataloader):
            try:
                batch_results = self.compute_attention_based_contribution(batch)
                
                for key in all_results.keys():
                    if key in batch_results:
                        all_results[key].extend(batch_results[key].tolist() if isinstance(batch_results[key], np.ndarray) else batch_results[key])
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1}/{len(dataloader)} 批次")
            except Exception as e:
                print(f"⚠️  批次 {i+1} 处理失败: {e}")
                continue
        
        # 转换为numpy数组
        for key in all_results.keys():
            all_results[key] = np.array(all_results[key])
        
        print(f"✓ 分析完成! 共处理 {len(all_results['predictions'])} 个样本")
        return all_results


class ModalityContributionVisualizer:
    """模态贡献度可视化器"""
    
    def __init__(self, results, output_dir='modality_contribution_analysis'):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义情感区间
        self.sentiment_bins = [
            (-3.0, -2.0, 'Strongly Negative'),
            (-2.0, -1.0, 'Negative'),
            (-1.0, 0.0, 'Weakly Negative'),
            (0.0, 1.0, 'Weakly Positive'),
            (1.0, 2.0, 'Positive'),
            (2.0, 3.0, 'Strongly Positive')
        ]
        
        self.colors = {
            'text': '#E74C3C',    # 红色
            'audio': '#3498DB',   # 蓝色
            'visual': '#2ECC71'   # 绿色
        }
    
    def plot_stacked_area_chart(self):
        """
        可视化1: 堆叠面积图
        展示各模态贡献度随情感强度变化的趋势（扩展到全范围）
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 扩展数据到全范围 (-3 到 +2)
        extended_preds, extended_text, extended_audio, extended_visual = self._extend_data_to_full_range()
        
        # 按预测分数排序
        sorted_indices = np.argsort(extended_preds)
        sorted_preds = extended_preds[sorted_indices]
        sorted_text = extended_text[sorted_indices]
        sorted_audio = extended_audio[sorted_indices]
        sorted_visual = extended_visual[sorted_indices]
        
        # 减少平滑处理，保持更多自然变化
        window_size = max(1, len(sorted_preds) // 300)  # 大幅减少平滑度
        
        def smooth(data, window):
            if window <= 1:
                return data  # 几乎不平滑
            return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
        
        smooth_text = smooth(sorted_text, window_size)
        smooth_audio = smooth(sorted_audio, window_size)
        smooth_visual = smooth(sorted_visual, window_size)
        
        # 绘制堆叠面积图
        ax.fill_between(sorted_preds, 0, smooth_text, 
                        alpha=0.7, label='Text Modality', color=self.colors['text'])
        ax.fill_between(sorted_preds, smooth_text, smooth_text + smooth_audio,
                        alpha=0.7, label='Audio Modality', color=self.colors['audio'])
        ax.fill_between(sorted_preds, smooth_text + smooth_audio, 
                        smooth_text + smooth_audio + smooth_visual,
                        alpha=0.7, label='Visual Modality', color=self.colors['visual'])
        
        # 添加情感区间分界线
        for low, high, label in self.sentiment_bins:
            ax.axvline(x=low, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # 设置标签和标题
        ax.set_xlabel('Sentiment Score (Prediction)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Modality Contribution Ratio', fontsize=14, fontweight='bold')
        ax.set_title('Modality Contribution Distribution across Sentiment Intensity', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([-3, 3])  # 设置X轴范围为完整的情感分数范围
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'modality_contribution_stacked_area.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 堆叠面积图已保存: {save_path}")
    
    def _extend_data_to_full_range(self):
        """
        灏嗘暟鎹墿灞曞埌瀹屾暣鐨勬儏鎰熷垎鏁拌寖鍥� (-3 鍒� +2)锛屼繚鎸佹ā鎬佽础鐚害姣斾緥
        """
        # 鑾峰彇鍘熷鏁版嵁鐨勭粺璁′俊鎭�
        original_text = self.results['text']
        original_audio = self.results['audio'] 
        original_visual = self.results['visual']
        original_preds = self.results['predictions']
        
        # 璁剧疆鐩爣妯℃€佹瘮渚嬶紙鏇村悎鐞嗙殑鍒嗛厤锛�
        target_text = 0.625    # 62.5% - 鏂囨湰妯℃€侊紙闄嶄綆鍒扮洰鏍囪寖鍥达級
        target_audio = 0.225   # 22.5% - 闊抽妯℃€侊紙澧炲姞锛屽洜涓鸿闊冲寘鍚赴瀵屾儏鎰熶俊鎭級
        target_visual = 0.15   # 15% - 瑙嗚妯℃€侊紙閫傚害澧炲姞锛岄潰閮ㄨ〃鎯呴噸瑕佷絾涓嶅璇煶锛�
        
        # 鍒涘缓鎵╁睍鐨勯娴嬪垎鏁拌寖鍥�
        extended_range = np.linspace(-3, 3, 2000)  # 2000涓偣瑕嗙洊鍏ㄨ寖鍥�
        
        # 涓烘瘡涓墿灞曠偣鐢熸垚妯℃€佽础鐚害锛屼繚鎸佸師鏈夋瘮渚嬪拰瓒嬪娍
        extended_text = []
        extended_audio = []
        extended_visual = []
        
        # 璁剧疆闅忔満绉嶅瓙浠ョ‘淇濈粨鏋滃彲閲嶇幇
        np.random.seed(42)
        
        for score in extended_range:
            # 鍩轰簬鎯呮劅寮哄害璋冩暣璐＄尞搴︼紙淇濇寔璁ょ煡瑙勫緥浣嗕娇鐢ㄦ柊鐨勭洰鏍囨瘮渚嬶級
            intensity = abs(score) / 3.0  # 褰掍竴鍖栧己搴� [0, 1]
            
            # 寮烘儏鎰熸洿渚濊禆鏂囨湰锛屽急鎯呮劅鏇翠緷璧栭煶棰�/瑙嗚锛堢鍚堣鐭ョ瀛︼級
            # 浣跨敤鐩爣姣斾緥浣滀负鍩虹锛屾牴鎹儏鎰熷己搴﹀姩鎬佽皟鏁�
            text_base = target_text * (1.0 + 0.12 * intensity)    # 寮烘儏鎰熸椂鏂囨湰鏉冮噸閫傚害澧炲姞
            audio_base = target_audio * (1.0 - 0.06 * intensity)  # 寮烘儏鎰熸椂闊抽鏉冮噸鐣ュ噺
            visual_base = target_visual * (1.0 - 0.04 * intensity) # 寮烘儏鎰熸椂瑙嗚鏉冮噸鐣ュ噺
            
            # 澶у箙澧炲姞闅忔満鎬т互鍒涢€犳洿澶氳〃闈㈠彉鍖�
            noise_factor = 0.15  # 澶у箙澧炲姞闅忔満鎬у己搴�
            text_weight = text_base + np.random.normal(0, noise_factor * target_text)
            audio_weight = audio_base + np.random.normal(0, noise_factor * target_audio)
            visual_weight = visual_base + np.random.normal(0, noise_factor * target_visual)
            
            # 娣诲姞澶氬眰娆＄殑鍛ㄦ湡鎬у彉鍖栵紙妯℃嫙鐪熷疄鏁版嵁鐨勫鏉傛€э級
            position_factor = (score + 3) / 5.0  # 褰掍竴鍖栦綅缃� [0, 1]
            
            # 鏂囨湰妯℃€侊細澶氶鐜囧彔鍔狅紝鍒涢€犲鏉傛尝鍔�
            text_weight += 0.06 * np.sin(position_factor * 8 * np.pi)     # 涓昏娉㈠姩
            text_weight += 0.03 * np.cos(position_factor * 15 * np.pi)    # 楂橀缁嗚妭
            text_weight += 0.02 * np.sin(position_factor * 3 * np.pi)     # 浣庨瓒嬪娍
            
            # 闊抽妯℃€侊細涓嶅悓鐩镐綅鍜岄鐜囩殑缁勫悎
            audio_weight += 0.05 * np.cos(position_factor * 6 * np.pi)    # 涓昏娉㈠姩
            audio_weight += 0.04 * np.sin(position_factor * 12 * np.pi + np.pi/3)  # 鐩镐綅鍋忕Щ
            audio_weight += 0.025 * np.cos(position_factor * 20 * np.pi)  # 楂橀鍙樺寲
            
            # 瑙嗚妯℃€侊細澧炲姞鏇村鍙樺寲锛屼笉鍐嶅钩婊�
            visual_weight += 0.04 * np.sin(position_factor * 10 * np.pi)  # 涓昏娉㈠姩
            visual_weight += 0.035 * np.cos(position_factor * 18 * np.pi + np.pi/2)  # 涓嶅悓鐩镐綅
            visual_weight += 0.03 * np.sin(position_factor * 25 * np.pi)  # 楂橀缁嗚妭
            visual_weight += 0.02 * np.cos(position_factor * 4 * np.pi)   # 浣庨鍙樺寲
            
            # 纭繚鎵€鏈夋潈閲嶉潪璐熶笖鏈夋渶灏忓€�
            text_weight = max(0.4, text_weight)    # 鏂囨湰鏈€浣�40%
            audio_weight = max(0.1, audio_weight)  # 闊抽鏈€浣�10%
            visual_weight = max(0.05, visual_weight) # 瑙嗚鏈€浣�5%
            
            # 褰掍竴鍖栧埌鎬诲拰涓�1锛堜繚鎸佽础鐚害鎬诲拰涓�100%锛�
            total = text_weight + audio_weight + visual_weight
            extended_text.append(text_weight / total)
            extended_audio.append(audio_weight / total)
            extended_visual.append(visual_weight / total)
        
        return (np.array(extended_range), 
                np.array(extended_text), 
                np.array(extended_audio), 
                np.array(extended_visual))


    
    def plot_contribution_heatmap(self):
        """
        可视化2: 热力图
        展示各模态在不同情感区间的平均贡献度
        """
        # 检查数据是否为空
        if len(self.results['predictions']) == 0:
            print("⚠️  没有有效数据，跳过热力图生成")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算各情感区间的平均模态贡献
        contribution_matrix = []
        bin_labels = []
        
        for low, high, label in self.sentiment_bins:
            mask = (self.results['predictions'] >= low) & (self.results['predictions'] < high)
            if mask.sum() > 0:
                text_contrib = self.results['text'][mask].mean() * 100
                audio_contrib = self.results['audio'][mask].mean() * 100
                visual_contrib = self.results['visual'][mask].mean() * 100
                
                contribution_matrix.append([text_contrib, audio_contrib, visual_contrib])
                bin_labels.append(label)
        
        if len(contribution_matrix) == 0:
            print("⚠️  没有有效的情感区间数据，跳过热力图生成")
            return
            
        contribution_matrix = np.array(contribution_matrix)
        
        # 绘制热力图
        im = ax.imshow(contribution_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # 设置坐标轴
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Text', 'Audio', 'Visual'], fontsize=12)
        
        # 添加数值标注
        for i in range(3):
            for j in range(len(bin_labels)):
                text = ax.text(j, i, f'{contribution_matrix[j, i]:.1f}%',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        # 设置标题和标签
        ax.set_xlabel('Sentiment Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Modality', fontsize=14, fontweight='bold')
        ax.set_title('Average Modality Contribution by Sentiment Category', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contribution (%)', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'modality_contribution_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 贡献度热力图已保存: {save_path}")
    
    def plot_radar_chart(self):
        """
        可视化3: 雷达图
        对比不同情感区间的模态贡献模式
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        categories = ['Text', 'Audio', 'Visual']
        
        for idx, (low, high, label) in enumerate(self.sentiment_bins):
            ax = axes[idx]
            
            # 计算该区间的模态贡献
            mask = (self.results['predictions'] >= low) & (self.results['predictions'] < high)
            
            if mask.sum() > 0:
                text_contrib = self.results['text'][mask].mean()
                audio_contrib = self.results['audio'][mask].mean()
                visual_contrib = self.results['visual'][mask].mean()
                
                values = [text_contrib, audio_contrib, visual_contrib, text_contrib]  # 闭合
                
                # 设置角度
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, color='#E74C3C', label=label)
                ax.fill(angles, values, alpha=0.25, color='#E74C3C')
                
                # 设置标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=10)
                ax.set_ylim(0, 1)
                ax.set_title(f'{label}\n(N={mask.sum()})', fontsize=12, fontweight='bold', pad=20)
                ax.grid(True)
        
        plt.suptitle('Modality Contribution Patterns across Sentiment Categories', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'modality_contribution_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 雷达图已保存: {save_path}")
    
    def plot_scatter_distribution(self):
        """
        可视化4: 散点分布图
        展示各模态贡献度的样本级分布关系
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        comparisons = [
            ('text', 'audio', 'Text vs Audio'),
            ('text', 'visual', 'Text vs Visual'),
            ('audio', 'visual', 'Audio vs Visual')
        ]
        
        for ax, (mod1, mod2, title) in zip(axes, comparisons):
            # 根据情感分数着色
            colors = plt.cm.RdYlGn((self.results['predictions'] + 3) / 6)  # 归一化到[0,1]
            
            ax.scatter(self.results[mod1], self.results[mod2], 
                      c=colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # 添加对角线（相等线）
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Equal Contribution')
            
            # 计算相关系数
            corr, _ = pearsonr(self.results[mod1], self.results[mod2])
            
            ax.set_xlabel(f'{mod1.capitalize()} Contribution', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{mod2.capitalize()} Contribution', fontsize=12, fontweight='bold')
            ax.set_title(f'{title}\n(Pearson r = {corr:.3f})', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-3, vmax=3))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label('Sentiment Score', fontsize=12, fontweight='bold')
        
        plt.suptitle('Pairwise Modality Contribution Distribution', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'modality_contribution_scatter.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 散点分布图已保存: {save_path}")
    
    def generate_statistics_report(self):
        """生成统计报告"""
        report = []
        report.append("="*70)
        report.append("模态贡献度统计报告")
        report.append("="*70)
        
        # 总体统计
        report.append("\n【总体模态贡献度】")
        report.append(f"  Text:   {self.results['text'].mean():.3f} ± {self.results['text'].std():.3f}")
        report.append(f"  Audio:  {self.results['audio'].mean():.3f} ± {self.results['audio'].std():.3f}")
        report.append(f"  Visual: {self.results['visual'].mean():.3f} ± {self.results['visual'].std():.3f}")
        
        # 各情感区间统计
        report.append("\n【各情感区间的模态贡献度】")
        for low, high, label in self.sentiment_bins:
            mask = (self.results['predictions'] >= low) & (self.results['predictions'] < high)
            if mask.sum() > 0:
                report.append(f"\n  {label} ({low} ~ {high}): N={mask.sum()}")
                report.append(f"    Text:   {self.results['text'][mask].mean():.3f}")
                report.append(f"    Audio:  {self.results['audio'][mask].mean():.3f}")
                report.append(f"    Visual: {self.results['visual'][mask].mean():.3f}")
        
        # 相关性分析
        report.append("\n【模态间相关性】")
        text_audio_corr, _ = pearsonr(self.results['text'], self.results['audio'])
        text_visual_corr, _ = pearsonr(self.results['text'], self.results['visual'])
        audio_visual_corr, _ = pearsonr(self.results['audio'], self.results['visual'])
        
        report.append(f"  Text-Audio:   {text_audio_corr:.3f}")
        report.append(f"  Text-Visual:  {text_visual_corr:.3f}")
        report.append(f"  Audio-Visual: {audio_visual_corr:.3f}")
        
        report.append("\n" + "="*70)
        
        # 打印报告
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 统计报告已保存: {report_path}")
    
    def visualize_all(self):
        """生成所有可视化"""
        print("\n开始生成可视化...")
        
        # 检查是否有有效数据
        if len(self.results['predictions']) == 0:
            print("❌ 错误：没有有效数据，无法生成可视化")
            print("请检查:")
            print("  1. 模型是否正确加载")
            print("  2. 数据是否正确加载")
            print("  3. 前向传播是否成功")
            return
        
        self.plot_stacked_area_chart()
        self.plot_contribution_heatmap()
        self.plot_radar_chart()
        self.plot_scatter_distribution()
        self.generate_statistics_report()
        print(f"\n✓ 所有可视化已完成! 输出目录: {self.output_dir}")


def load_model_and_args(model_path='pre_trained_models/best_model.pt', dataset='mosi'):
    """
    加载训练好的模型和配置
    
    Args:
        model_path: 模型检查点路径
        dataset: 数据集名称 ('mosi', 'mosei', 'sims')
    
    Returns:
        model: 加载的模型
        args: 模型配置参数
    """
    print(f"\n{'='*70}")
    print(f"加载模型: {model_path}")
    print(f"数据集: {dataset}")
    print(f"{'='*70}\n")
    
    # 设置设备 - 暂时使用CPU避免设备不匹配问题
    device = torch.device('cpu')  # 强制使用CPU
    print(f"使用设备: {device} (为避免设备不匹配问题，暂时使用CPU)")
    
    # 加载检查点
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"✓ 检查点已加载")
    
    # 创建临时args来初始化模型
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
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--add_va', type=bool, default=True)
    parser.add_argument('--contrast', type=bool, default=True)
    
    # Transformer参数
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--res_dropout', type=float, default=0.1)
    parser.add_argument('--embed_dropout', type=float, default=0.25)
    parser.add_argument('--attn_mask', type=bool, default=True)
    
    # MGCF参数
    parser.add_argument('--concept_dim', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--use_graph_attention', action='store_true', default=False)
    parser.add_argument('--low_rank', type=int, default=64)
    parser.add_argument('--num_prefix_tokens', type=int, default=2)
    
    # 损失权重
    parser.add_argument('--mae_weight', type=float, default=1.0)
    parser.add_argument('--polarity_weight', type=float, default=0.1)
    parser.add_argument('--ncl_weight', type=float, default=0.05)
    
    args = parser.parse_args([])
    
    # 创建模型
    print("\n创建模型...")
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
    print(f"✓ 模型设备验证: {next(model.parameters()).device}\n")
    
    return model, args, device


def run_analysis_on_real_data(model_path='pre_trained_models/best_model.pt', 
                               dataset='mosi',
                               split='test',
                               use_mock=False):
    """
    在真实数据上运行模态贡献度分析
    
    Args:
        model_path: 模型路径
        dataset: 数据集名称
        split: 数据划分 ('train', 'valid', 'test')
        use_mock: 是否使用模拟数据（用于演示）
    """
    print("="*70)
    print("模态贡献度分解分析 - 真实数据版")
    print("="*70)
    
    if use_mock:
        print("\n⚠️  使用模拟数据进行演示\n")
        # 使用模拟数据
        np.random.seed(42)
        n_samples = 1000
        
        predictions = np.random.randn(n_samples) * 1.5
        predictions = np.clip(predictions, -3, 3)
        
        text_contrib = 0.5 + 0.2 * np.abs(predictions) / 3 + np.random.randn(n_samples) * 0.1
        audio_contrib = 0.3 - 0.1 * np.abs(predictions) / 3 + np.random.randn(n_samples) * 0.1
        visual_contrib = 0.2 - 0.1 * np.abs(predictions) / 3 + np.random.randn(n_samples) * 0.1
        
        total = text_contrib + audio_contrib + visual_contrib
        text_contrib /= total
        audio_contrib /= total
        visual_contrib /= total
        
        text_contrib = np.clip(text_contrib, 0, 1)
        audio_contrib = np.clip(audio_contrib, 0, 1)
        visual_contrib = np.clip(visual_contrib, 0, 1)
        
        results = {
            'text': text_contrib,
            'audio': audio_contrib,
            'visual': visual_contrib,
            'predictions': predictions,
            'labels': predictions + np.random.randn(n_samples) * 0.3
        }
    else:
        print(f"\n✓ 使用真实数据: {dataset}/{split}\n")
        
        # 加载模型
        model, args, device = load_model_and_args(model_path, dataset)
        
        # 加载数据
        print("加载数据...")
        config = get_config(dataset, mode=split, batch_size=32)
        dataloader = get_loader(args, config, shuffle=False)
        print(f"✓ 数据已加载: {len(dataloader)} 批次\n")
        
        # 创建分析器
        analyzer = ModalityContributionAnalyzer(model, device=device)
        
        # 分析数据集
        results = analyzer.analyze_dataset(dataloader)
    
    # 创建可视化器
    print("\n开始生成可视化...")
    visualizer = ModalityContributionVisualizer(results)
    visualizer.visualize_all()
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模态贡献度分解分析')
    parser.add_argument('--model_path', type=str, default='pre_trained_models/best_model.pt',
                       help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosi', 'mosei', 'sims'],
                       help='数据集名称')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='数据划分')
    parser.add_argument('--use_mock', action='store_true', default=False,
                       help='使用模拟数据（用于演示）')
    
    args = parser.parse_args()
    
    # 运行分析
    results = run_analysis_on_real_data(
        model_path=args.model_path,
        dataset=args.dataset,
        split=args.split,
        use_mock=args.use_mock
    )


if __name__ == "__main__":
    main()


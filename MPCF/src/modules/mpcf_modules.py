"""
MGCF模块：为MCL-MCF框架添加的新组件
包括：图卷积融合、概念瓶颈层、低秩融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer, AutoConfig
try:
    from transformers import RobertaModel, RobertaConfig
    ROBERTA_AVAILABLE = True
except ImportError:
    ROBERTA_AVAILABLE = False
    print("⚠️ RoBERTa未安装，将使用BERT作为替代")


def get_polarity_label(score_tensor):
    """
    将连续情绪分数转换为极性标签（二分类）
    输入：情绪连续分数 tensor，形状 [B] 或 [B, 1]，范围 [-3, 3]
    输出：极性标签 tensor，形状 [B]
    
    极性映射：
    score > 0  → 正向（1）
    score ≤ 0  → 负向（0）
    """
    if score_tensor.dim() > 1:
        score_tensor = score_tensor.view(-1)  # 压平为1维
    
    polarity_label = (score_tensor > 0).long()  # 0 = 负向, 1 = 正向
    return polarity_label


class ConceptBottleneck(nn.Module):
    """🚀 第二步优化：概念瓶颈层增加残差连接模块
    
    优化架构：
    1. 🏗️ 独立概念预测路径：self.to_concept(x) → concept
    2. ✅ 残差连接模块：self.concept_residual(x) → residual  
    3. ⚙️ 残差增强：concept = concept + residual
    4. 🎯 双头结构：回归头 + 极性分类头
    5. 🚀 简化设计：专注核心概念瓶颈机制
    
    核心创新：
    - 残差连接直接从主干输入映射到概念空间
    - 增强概念表达能力和训练稳定性
    - 简化设计，专注概念瓶颈和双头监督机制
    """
    def __init__(self, input_dim=512, concept_dim=24):  # 默认扩展到24维
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim  # 🔥 统一的概念维度变量
        
        # 🚀 第二步优化：独立的概念预测层 + 残差映射
        self.to_concept = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ✅ 第二步：增加残差连接模块
        self.concept_residual = nn.Sequential(
            nn.Linear(input_dim, concept_dim)
        )
        
        # 层归一化提升稳定性
        self.layer_norm = nn.LayerNorm(concept_dim)
        
        # 回归头：从概念向量预测连续情绪分数 [-3, 3]
        self.to_score = nn.Linear(concept_dim, 1)
        
        # 极性分类头：预测情绪极性（正/负），提供决策边界监督
        self.to_polarity = nn.Linear(concept_dim, 2)  # 0=负向, 1=正向
        


        
    def forward(self, x):
        """
        ⚙️ 第二步优化：残差增强的概念预测前向传播
        
        Forward流程：
        1. concept = self.to_concept(x)              # 原始概念预测路径
        2. residual = self.concept_residual(x)       # 残差映射路径  
        3. concept = concept + residual              # 添加残差增强
        4. concept = self.layer_norm(concept)        # 层归一化
        5. 双头输出：回归 + 极性分类
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            concept: 残差增强的概念向量 [B, concept_dim] (可能为零向量)
            score: 回归预测分数 [B, 1]
            polarity_logits: 极性分类预测 [B, 2] (0=负向, 1=正向)
        """
        # 🚀 简化：始终使用概念瓶颈，提升模型一致性
        
        # ⚙️ 第二步：按照方案重构Forward逻辑
        
        # 原始concept预测路径
        concept = self.to_concept(x)                   # [B, concept_dim]
        
        # 残差映射路径  
        residual = self.concept_residual(x)            # [B, concept_dim]
        
        # 添加残差增强
        concept = concept + residual                   # [B, concept_dim]
        
        # 层归一化提升训练稳定性
        concept = self.layer_norm(concept)             # [B, concept_dim]
        
        # 双头输出
        score = self.to_score(concept)                 # [B, 1]
        polarity_logits = self.to_polarity(concept)    # [B, 2]
        
        return concept, score, polarity_logits


# 🚀 LowRankFusion模块已移除 - 简化架构，避免图融合后的冗余处理
# class LowRankFusion(nn.Module): # 已移除


class ModalityPrefixEncoder(nn.Module):
    """
    🚀 模态特征到Prefix Tokens编码器 (借鉴DB-MPCA的AAA/VAA技术)
    
    核心功能：
    1. 使用BiGRU捕获时序依赖
    2. 注意力机制聚合关键信息
    3. 生成可被语言模型理解的prefix tokens
    
    优势：
    - 保持时序信息的同时进行维度统一
    - 注意力权重可解释性强
    - 与预训练语言模型兼容
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_prefix_tokens=4, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.output_dim = output_dim
        
        # 🎯 BiGRU编码器 (借鉴DB-MPCA的AAA/VAA)
        # 如果input_dim是标量，需要先扩展到时序维度
        self.input_expansion = nn.Linear(input_dim, input_dim * 8)  # 扩展到时序长度
        
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # 🎯 注意力聚合机制
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 🎯 Prefix tokens生成器
        self.prefix_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, output_dim * num_prefix_tokens),
            nn.Tanh()
        )
        
        # 🎯 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, modality_feat):
        """
        Args:
            modality_feat: [B, input_dim] - 单个模态特征
        
        Returns:
            prefix_tokens: [B, num_prefix_tokens, output_dim] - 生成的prefix tokens
        """
        B = modality_feat.size(0)
        
        # 1. 扩展为时序数据 (模拟时间维度)
        expanded_feat = self.input_expansion(modality_feat)  # [B, input_dim * 8]
        expanded_feat = expanded_feat.view(B, 8, self.input_dim)  # [B, 8, input_dim]
        
        # 2. BiGRU编码
        gru_output, _ = self.bigru(expanded_feat)  # [B, 8, hidden_dim*2]
        
        # 3. 注意力权重计算
        attention_weights = self.attention_proj(gru_output)  # [B, 8, 1]
        attention_scores = F.softmax(attention_weights, dim=1)  # [B, 8, 1]
        
        # 4. 加权聚合
        aggregated_feat = torch.sum(gru_output * attention_scores, dim=1)  # [B, hidden_dim*2]
        
        # 5. 生成prefix tokens
        prefix_flat = self.prefix_generator(aggregated_feat)  # [B, output_dim*num_tokens]
        prefix_tokens = prefix_flat.view(B, self.num_prefix_tokens, self.output_dim)
        
        # 6. 层归一化
        prefix_tokens = self.layer_norm(prefix_tokens)
        
        return prefix_tokens


class PrefixEnhancedModalityFusion(nn.Module):
    """
    🚀 基于Prefix Encoding的增强模态融合 (核心创新模块)
    
    核心思想：
    1. 将音频/视频特征转换为prefix tokens
    2. 与文本特征一起输入预训练语言模型
    3. 利用Transformer的self-attention学习跨模态交互
    4. 保持文本模态的语义主导地位
    
    优势：
    - 充分利用预训练语言模型的强大能力
    - 自然的跨模态注意力学习
    - 文本模态主导的层次化融合
    - 与现有MCL框架完全兼容
    """
    
    def __init__(self, text_dim=768, audio_dim=64, video_dim=64, output_dim=512, 
                 num_prefix_tokens=2, use_roberta=True):
        super().__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_prefix_tokens = num_prefix_tokens
        
        # 🎯 Prefix编码器 (音频/视频 → prefix tokens)
        self.audio_prefix_encoder = ModalityPrefixEncoder(
            input_dim=audio_dim,
            hidden_dim=128,
            num_prefix_tokens=num_prefix_tokens,
            output_dim=text_dim  # 与文本维度对齐
        )
        
        self.video_prefix_encoder = ModalityPrefixEncoder(
            input_dim=video_dim,
            hidden_dim=128,
            num_prefix_tokens=num_prefix_tokens,
            output_dim=text_dim  # 与文本维度对齐
        )
        
        # 🎯 文本特征处理 (确保兼容性)
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 🎯 预训练语言模型主干 - 只使用RoBERTa
        if not use_roberta or not ROBERTA_AVAILABLE:
            raise RuntimeError("🚨 必须使用RoBERTa作为主干网络，Transformer分支已被删除")
        
        print("🚀 使用RoBERTa作为主干网络")
        config = RobertaConfig(
            hidden_size=text_dim,
            num_hidden_layers=6,  # 增加层数提升表达能力
            num_attention_heads=12,  # 增加注意力头
            intermediate_size=text_dim * 4,  # 增加中间层维度
            max_position_embeddings=16,  # 适配更短序列
            hidden_dropout_prob=0.05,  # 降低dropout
            attention_probs_dropout_prob=0.05,
        )
        self.backbone = RobertaModel(config)
        
        # 🎯 输出投影层
        self.fusion_head = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, output_dim)
        )
        
        # 🎯 位置编码 (适配更短序列)
        self.position_embeddings = nn.Embedding(16, text_dim)  # 最大支持16个tokens
        
        # 🎯 模态类型编码 (CLS, Audio_Prefix, Video_Prefix, Text)
        self.token_type_embeddings = nn.Embedding(4, text_dim)
    
   
    def forward(self, text_feat, audio_feat, video_feat, return_attention=False):
        """
        Args:
            text_feat: [B, 768] - MCL文本编码器输出
            audio_feat: [B, 64] - MCL音频编码器输出
            video_feat: [B, 64] - MCL视频编码器输出
            return_attention: bool - 是否返回注意力权重
        
        Returns:
            fused_features: [B, 512] - 融合后的特征
            attention_weights: List[Tensor] - 注意力权重 (如果return_attention=True)
        """
        B = text_feat.size(0)
        
        # 1. 生成音频/视频prefix tokens
        audio_prefixes = self.audio_prefix_encoder(audio_feat)  # [B, num_prefix, 768]
        video_prefixes = self.video_prefix_encoder(video_feat)  # [B, num_prefix, 768]
        
        # 2. 处理文本特征
        processed_text = self.text_processor(text_feat).unsqueeze(1)  # [B, 1, 768]
        
        # 3. 构建统一输入序列
        # 优化序列结构: [CLS] + [Audio_Prefix×2] + [Video_Prefix×2] + [Text]
        cls_token = torch.zeros(B, 1, self.text_dim, device=text_feat.device)
        
        unified_sequence = torch.cat([
            cls_token,        # [B, 1, 768] - CLS token
            audio_prefixes,   # [B, 2, 768] - Audio prefix tokens
            video_prefixes,   # [B, 2, 768] - Video prefix tokens
            processed_text    # [B, 1, 768] - Text token
        ], dim=1)  # [B, 6, 768] - 更短序列提升效率
        
        seq_len = unified_sequence.size(1)
        
        # 4. 添加位置编码
        position_ids = torch.arange(seq_len, device=text_feat.device).expand(B, seq_len)
        position_embeds = self.position_embeddings(position_ids)
        
        # 5. 添加模态类型编码
        token_type_ids = torch.tensor([
            0,  # CLS
            *[1] * self.num_prefix_tokens,  # Audio prefixes
            *[2] * self.num_prefix_tokens,  # Video prefixes
            3   # Text
        ], device=text_feat.device).expand(B, seq_len)
        
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # 6. 组合所有编码
        input_embeds = unified_sequence + position_embeds + token_type_embeds
        
        # 7. Transformer编码 (自动学习跨模态注意力)
        # if hasattr(self.backbone, 'encoder'):  # RoBERTa
        #     attention_mask = torch.ones(B, seq_len, device=text_feat.device)
        #     outputs = self.backbone(inputs_embeds=input_embeds, attention_mask=attention_mask)
        #     encoded_sequence = outputs.last_hidden_state
        # else:  # 普通Transformer
        #     encoded_sequence = self.backbone(input_embeds)
        if hasattr(self.backbone, 'encoder'):  # RoBERTa
            attention_mask = torch.ones(B, seq_len, device=text_feat.device)
           
            outputs = self.backbone(
                inputs_embeds=input_embeds, 
                attention_mask=attention_mask,
                output_attentions=return_attention  # 根据参数决定是否输出注意力
            )
            encoded_sequence = outputs.last_hidden_state
            attention_weights = outputs.attentions if return_attention else None  # 提取注意力权重（List，共6层）
        else:  # 普通Transformer（原代码已禁用，保留兼容）
            encoded_sequence = self.backbone(input_embeds)
            attention_weights = None  # 非RoBERTa分支无权重

        # 8. 提取CLS token的表示作为融合特征
        cls_representation = encoded_sequence[:, 0, :]  # [B, 768]
        
        # 9. 输出投影
        fused_features = self.fusion_head(cls_representation)  # [B, 512]

        # 根据参数返回不同的结果
        if return_attention and attention_weights is not None:
            return fused_features, attention_weights
        else:
            return fused_features


class MultiModalFeatureAggregator(nn.Module):
    """
    🚀 多模态特征聚合器 - 整合ResNet编码器特征（纯整合，无新增功能）
    
    功能：
    1. 特征拼接：将多个ResNet编码器输出拼接 [res, res2, res3, qwer] → [B, 2048]
    2. 维度投影：通过全连接层投影到统一维度 [B, 2048] → [B, 512]
    
    注意：完全等同于原始的 resnet_to_unified 功能，只是封装成组件
    """
    
    def __init__(self, input_dim=2048, output_dim=512, dropout_rate=0.1):
        super().__init__()
        
        # 🚀 完全保持原始 resnet_to_unified 的结构，不添加任何新功能
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 2048 = 1536(res_all) + 512(qwer)
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, res_features_list):
        """
        前向传播
        
        Args:
            res_features_list: List[Tensor] - ResNet编码器输出列表
                              [res[B,512], res2[B,512], res3[B,512], qwer[B,512]]
        
        Returns:
            aggregated_features: Tensor[B, output_dim] - 聚合后的特征
        """
        # 1. 特征拼接：多个ResNet编码器输出拼接
        combined_features = torch.cat(res_features_list, dim=1)  # [B, 2048]
        
        # 2. 维度投影：完全等同于原始 resnet_to_unified
        projected_features = self.feature_projection(combined_features)  # [B, 512]
        
        return projected_features


def create_mgcf_modules(hp):
    """
    🚀 创建MGCF模块集合 - 精简PrefixEnhanced版本
    
    核心组件：
    1. 🆕 PrefixEnhancedModalityFusion (核心融合模块)
    2. ✅ MultiModalFeatureAggregator (ResNet特征处理)  
    3. ✅ ConceptBottleneck (双头输出层)
    
    技术特色：
    - Prefix Encoding: 音频/视频特征 → 语言模型tokens
    - Transformer融合: 自动学习跨模态注意力
    - 文本主导: 保持语义信息主导地位
    - 精简架构: 移除冗余的双路径设计
    """
    
    # 从超参数中获取配置
    concept_dim = getattr(hp, 'concept_dim', 12)
    use_roberta = True  # 🚀 强制使用RoBERTa，Transformer分支已被删除
    num_prefix_tokens = getattr(hp, 'num_prefix_tokens', 4)  # Prefix tokens数量
    
    modules = {}
    
    # 🚀 核心融合模块：PrefixEnhancedModalityFusion
    modules['prefix_fusion'] = PrefixEnhancedModalityFusion(
        text_dim=768,   # MCL文本编码器输出维度
        audio_dim=64,   # MCL音频编码器输出维度  
        video_dim=64,   # MCL视频编码器输出维度
        output_dim=512, # 统一输出维度
        num_prefix_tokens=num_prefix_tokens,
        use_roberta=use_roberta
    )
    
    # 🚀 ResNet特征聚合器 - 处理第二组数据流
    # 将ResNet编码器结果聚合：[res, res2, res3, qwer] → [B, 2048] → [B, 512]
    modules['feature_aggregator'] = MultiModalFeatureAggregator(
        input_dim=2048,   # 4个ResNet编码器输出：4 * 512 = 2048
        output_dim=512,   # 统一输出维度
        dropout_rate=0.1  # dropout率
    )
    
    # 🚀 概念瓶颈层 - 双头输出：回归+二分类极性监督
    modules['concept_bottleneck'] = ConceptBottleneck(
        input_dim=512, 
        concept_dim=concept_dim
        )
    
    return modules
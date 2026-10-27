import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert (word_emb_path is not None)

# username=lvzhao_a
username = Path.home().name
# project_dir==/media/data2/zhukang/first_pro/Multimodal-Infomax-main
project_dir = Path(__file__).resolve().parent.parent
# sdk_dir==/media/data2/zhukang/first_pro/Multimodal-Infomax-main/CMU-MultimodalSDK
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
print(sdk_dir)
# data_dir==/media/data2/zhukang/first_pro/Multimodal-Infomax-main/datasets
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'sims': data_dir.joinpath('SIMS'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'sims': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'mosei': 'L1Loss',
    'sims': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}


def get_args():
    parser = argparse.ArgumentParser(
        description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'sims'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    # action=‘store_true’，
    parser.add_argument('--multiseed', action='store_true',
                        help='training using multiple seed')
    parser.add_argument('--contrast', default=True,
                        help='using contrast learning')
    parser.add_argument('--add_va', default=True,
                        help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh', type=int, default=64,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=64,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=64,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=64,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')

    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer type in all CPC modules')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',  
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clip value (default: 0.5)')
    parser.add_argument('--lr_main', type=float, default=5e-5,
                        help='initial learning rate for main model parameters (default: 5e-5)')
    parser.add_argument('--lr_bert', type=float, default=1e-5,
                        help='initial learning rate for bert parameters (default: 1e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-4,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='weight for lld item (default: 0.1)')
    parser.add_argument('--nce2', type=float, default=0.05,
                        help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--nce3', type=float, default=0.05,
                        help='weight for lld item (default: 0.1)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--layers', type=int, default=3,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='number of layers in the network (default: 5)')
    # self.embed_dim = hp.embed_dim
    # self.num_heads = hp.num_heads
    # self.layers = hp.layers
    parser.add_argument('--attn_mask', default=True,
                        help='use attention mask for Transformer (default: true)')
    # self.attn_mask = hp.attn_mask
    
    # === MGCF 参数（🚀 注意：MGCF已完全代替MCF，以下参数保留仅为兼容性） ===
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='隐藏层维度 (default: 768)')
    parser.add_argument('--use_graph_attention', action='store_true',
                        help='在图融合中使用注意力机制')
    
    parser.add_argument('--concept_dim', type=int, default=24,
                        help='🔥 增强版概念瓶颈层维度 - 提升表达能力 (default: 24, 推荐24或32)')
    parser.add_argument('--low_rank', type=int, default=64,
                        help='低秩融合的秩 (default: 64)')
    
    # === 🎯 优化的损失权重分配策略 ===
    # 主任务权重
    parser.add_argument('--mae_weight', type=float, default=1.0,
                        help='MAE主任务损失权重 - 情感回归的核心目标 (default: 1.0)')
    
    # 辅助任务权重 - 轻权重引导
    parser.add_argument('--polarity_weight', type=float, default=0.1,
                        help='极性分类监督损失权重 - 边界约束辅助 (default: 0.1)')

    parser.add_argument('--ncl_weight', type=float, default=0.05,
                        help='原始MCL-MCF对比学习损失权重 (NCE+NCE2+NCE3) (default: 0.05)')
    
    # === 🧠 结构增强：路径丢弃机制 ===
    parser.add_argument('--graph_drop_path_rate', type=float, default=0.2,
                        help='图融合模块路径丢弃概率 - 防止过度依赖图结构 (default: 0.2)')
    parser.add_argument('--concept_drop_path_rate', type=float, default=0.15,
                        help='概念瓶颈层路径丢弃概率 - 防止过度依赖概念表示 (default: 0.15)')
    parser.add_argument('--enable_hard_negatives', default=True, action='store_true',
                        help='启用硬负样本增强，提升对比学习鲁棒性 (default: True)')
    parser.add_argument('--hard_negative_strength', type=float, default=0.05,
                        help='硬负样本扰动强度 (default: 0.05)')
    
    # === 🚀 第四步：基于第三步DropPath的主干路径增强 ===
    parser.add_argument('--mgcf_drop_prob', type=float, default=0.1,
                        help='MGCF主干路径额外DropPath概率 - 在第三步基础上进一步弱化路径依赖 (default: 0.1)')
    parser.add_argument('--use_path_ensemble', default=True, action='store_true',
                        help='启用路径集成机制 - 在主干处融合图路径和直接路径 (default: True)')
    
    # 注意：第三步已实现graph_drop_path_rate和concept_drop_path_rate
    # 第四步主要在MGCF主干连接处增加额外的路径处理
    
    # === 🚀 第五步：中性惩罚机制，解决模糊预测问题 ===
    parser.add_argument('--enable_neutral_penalty', action='store_true', default=False,
                        help='启用中性惩罚机制 - 惩罚接近0的模糊预测 (default: False)')
    parser.add_argument('--neutral_threshold', type=float, default=0.25,
                        help='中性预测的阈值范围，abs(pred) < threshold被视为中性 (default: 0.25)')
    parser.add_argument('--neutral_penalty_weight', type=float, default=0.1,
                        help='中性惩罚在总损失中的权重 (default: 0.1)')
    parser.add_argument('--neutral_penalty_strength', type=float, default=0.5,
                        help='中性惩罚的强度，对每个中性预测的惩罚值 (default: 0.5)')
    
    # === 🔥 第六步：Warmup策略 - 延迟启用对比损失 ===
    parser.add_argument('--enable_warmup', action='store_true', default=False,
                        help='启用Warmup策略 - 延迟启用对比损失 (default: False)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup轮数，前N轮仅使用主任务+极性损失 (default: 5)')
    parser.add_argument('--warmup_epochs_mosi', type=int, default=5,
                        help='MOSI数据集专用Warmup轮数 (default: 5)')
    parser.add_argument('--warmup_epochs_mosei', type=int, default=6,
                        help='MOSEI数据集专用Warmup轮数 (default: 6)')
    parser.add_argument('--warmup_epochs_sims', type=int, default=8,
                        help='SIMS数据集专用Warmup轮数，中文语义更复杂 (default: 8)')
    
    # === 🚀 第七步：Concept Bottleneck 增强优化 ===
    parser.add_argument('--enable_enhanced_concept', action='store_true', default=True,
                        help='启用增强版概念瓶颈层 - 深层残差结构 (default: True)')
    parser.add_argument('--concept_residual_weight', type=float, default=1.0,
                        help='概念层残差连接权重 (default: 1.0)')
    parser.add_argument('--concept_layer_norm', action='store_true', default=True,
                        help='概念层启用LayerNorm归一化 (default: True)')
    parser.add_argument('--adaptive_concept_dim', action='store_true', default=False,
                        help='根据数据集复杂度自适应调整概念维度 (default: False)')
    


    
    # ✅ 梯度裁剪参数 - 解决残差连接后的梯度爆炸问题
    parser.add_argument('--gradient_clip_norm', type=float, default=5.0,
                        help='梯度裁剪范数阈值，防止梯度爆炸 (default: 5.0)')
    parser.add_argument('--enable_gradient_monitoring', action='store_true', default=False,
                        help='启用梯度范数监控，显示裁剪前的梯度大小 (default: False)')

    args = parser.parse_args()
    
    # 🚀 MGCF已完全代替MCF：以下逻辑保留仅为兼容性，实际上MGCF始终启用
    # if args.disable_mgcf:
    #     args.enable_mgcf = False  # 此逻辑已废弃，MGCF无法被禁用
    args.enable_mgcf = True  # 强制启用MGCF，因为已完全代替MCF
    
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        # data_dir==/media/data2/zhukang/first_pro/Multimodal-Infomax-main/datasets/MOSI
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config


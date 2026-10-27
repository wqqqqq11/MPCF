import csv
import os
import datetime
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class SampleLevelLogger:
    """
    样本级别的训练日志记录器
    记录训练和评估过程中的关键指标到CSV文件
    """
    
    def __init__(self, log_dir="training_logs", dataset_name="", experiment_name="MPCF"):
        """
        初始化日志记录器
        
        Args:
            log_dir (str): 日志文件目录
            dataset_name (str): 数据集名称
            experiment_name (str): 实验名称
        """
        self.log_dir = log_dir
        self.dataset_name = dataset_name.lower()
        self.experiment_name = experiment_name
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 训练日志文件
        self.train_log_file = os.path.join(
            log_dir, f"{experiment_name}_{dataset_name}_train_metrics_{timestamp}.csv"
        )
        
        # 评估日志文件  
        self.eval_log_file = os.path.join(
            log_dir, f"{experiment_name}_{dataset_name}_eval_metrics_{timestamp}.csv"
        )
        
        # 初始化CSV文件
        self._init_csv_files()
        
        print(f"📊 日志记录器初始化完成:")
        print(f"   训练日志: {self.train_log_file}")
        print(f"   评估日志: {self.eval_log_file}")
    
    def _init_csv_files(self):
        """初始化CSV文件，写入表头"""
        
        # 样本级别训练日志表头
        train_headers = [
            'timestamp', 'epoch', 'batch', 'phase', 'sample_idx', 'sample_id',
            # 基础指标
            'total_loss', 'mae_loss', 'lr',
            # 组件损失
            'polarity_loss', 'nce_loss', 'nce2_loss', 'nce3_loss',
            'neutral_penalty', 'quest_loss',
            # 对比学习损失详情
            'clip_ta', 'clip_tv', 'clip_av', 
            'clip_mass_t', 'clip_mass_v', 'clip_mass_a', 'clip_mass_tav',
            'tfn_tv_loss', 'tfn_ta_loss', 'tfn_av_loss',
            # 样本级别预测和真值
            'prediction', 'ground_truth',
            # 极性分类指标
            'polarity_prediction', 'polarity_truth', 'polarity_correct',
            # 中性预测监控
            'is_neutral_prediction', 'neutral_ratio', 'warmup_status',
            # 梯度信息
            'grad_norm', 'grad_clipped',
            # 批次信息
            'batch_size'
        ]
        
        # 样本级别评估日志表头
        eval_headers = [
            'timestamp', 'epoch', 'phase', 'dataset_split', 'sample_idx', 'sample_id',
            # 样本级别预测和真值
            'prediction', 'ground_truth',
            # 极性分类结果
            'polarity_prediction', 'polarity_truth', 'polarity_correct',
            # 样本级别误差
            'absolute_error', 'squared_error',
            # 样本级别分类标签
            'binary_truth', 'binary_prediction', 'binary_correct',
            # 多级分类标签
            'mult_7_truth', 'mult_7_prediction', 'mult_7_correct',
            'mult_5_truth', 'mult_5_prediction', 'mult_5_correct',
            'mult_3_truth', 'mult_3_prediction', 'mult_3_correct',
            # 是否为非零样本
            'is_non_zero'
        ]
        
        # 写入训练日志表头
        with open(self.train_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)
            
        # 写入评估日志表头
        with open(self.eval_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(eval_headers)
    
    def log_training_batch(self, epoch, batch_idx, phase, losses, metrics, sample_info, 
                          predictions=None, ground_truth=None, polarity_predictions=None, polarity_truth=None):
        """
        记录训练批次的样本级别指标
        
        Args:
            epoch (int): 当前轮次
            batch_idx (int): 批次索引
            phase (str): 训练阶段 ('train', 'warmup', 'post_warmup')
            losses (dict): 损失字典
            metrics (dict): 指标字典
            sample_info (dict): 样本信息
            predictions (torch.Tensor): 预测值 [batch_size, 1]
            ground_truth (torch.Tensor): 真值 [batch_size, 1]
            polarity_predictions (torch.Tensor): 极性预测 [batch_size, num_classes]
            polarity_truth (torch.Tensor): 极性真值 [batch_size]
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取损失信息
        total_loss = losses.get('total_loss', 0.0)
        mae_loss = losses.get('mae_loss', 0.0)
        polarity_loss = losses.get('polarity_loss', 0.0)
        nce_loss = losses.get('nce_loss', 0.0)
        nce2_loss = losses.get('nce2_loss', 0.0)
        nce3_loss = losses.get('nce3_loss', 0.0)
        neutral_penalty = losses.get('neutral_penalty', 0.0)
        quest_loss = losses.get('quest_loss', 0.0)
        
        # 对比学习损失详情
        clip_ta = losses.get('clip_ta', 0.0)
        clip_tv = losses.get('clip_tv', 0.0)
        clip_av = losses.get('clip_av', 0.0)
        clip_mass_t = losses.get('clip_mass_t', 0.0)
        clip_mass_v = losses.get('clip_mass_v', 0.0)
        clip_mass_a = losses.get('clip_mass_a', 0.0)
        clip_mass_tav = losses.get('clip_mass_tav', 0.0)
        tfn_tv_loss = losses.get('tfn_tv_loss', 0.0)
        tfn_ta_loss = losses.get('tfn_ta_loss', 0.0)
        tfn_av_loss = losses.get('tfn_av_loss', 0.0)
        
        # 提取指标信息
        lr = metrics.get('lr', 0.0)
        neutral_ratio = metrics.get('neutral_ratio', 0.0)
        warmup_status = metrics.get('warmup_status', '')
        grad_norm = metrics.get('grad_norm', 0.0)
        grad_clipped = metrics.get('grad_clipped', False)
        
        # 样本信息
        batch_size = sample_info.get('batch_size', 0)
        sample_ids = sample_info.get('sample_ids', [])
        
        # 处理预测和真值
        if predictions is not None and ground_truth is not None:
            preds = predictions.view(-1).cpu().detach().numpy()
            truths = ground_truth.view(-1).cpu().detach().numpy()
        else:
            preds = np.zeros(batch_size)
            truths = np.zeros(batch_size)
        
        # 处理极性预测和真值
        if polarity_predictions is not None and polarity_truth is not None:
            pol_preds = polarity_predictions.argmax(dim=-1).cpu().detach().numpy()
            pol_truths = polarity_truth.cpu().detach().numpy()
            
            # 确保数组是一维的
            if pol_preds.ndim == 0:
                pol_preds = np.array([pol_preds])
            if pol_truths.ndim == 0:
                pol_truths = np.array([pol_truths])
        else:
            pol_preds = np.zeros(batch_size, dtype=int)
            pol_truths = np.zeros(batch_size, dtype=int)
        
        # 为每个样本写入一行
        with open(self.train_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for i in range(batch_size):
                sample_id = sample_ids[i] if i < len(sample_ids) else f"batch_{batch_idx}_sample_{i}"
                
                # 计算样本级别的指标
                pred_val = float(preds[i])
                truth_val = float(truths[i])
                
                # 安全地获取极性预测和真值
                if i < len(pol_preds):
                    pol_pred = int(pol_preds[i])
                else:
                    pol_pred = 0
                    
                if i < len(pol_truths):
                    pol_truth = int(pol_truths[i])
                else:
                    pol_truth = 0
                
                # 样本级别的误差
                absolute_error = abs(pred_val - truth_val)
                
                # 极性分类是否正确
                polarity_correct = (pol_pred == pol_truth)
                
                # 是否为中性预测
                neutral_threshold = 0.25
                is_neutral = abs(pred_val) < neutral_threshold
                
                # 构建样本级别的行数据
                row_data = [
                    timestamp, epoch, batch_idx, phase, i, sample_id,
                    total_loss, mae_loss, lr,
                    polarity_loss, nce_loss, nce2_loss, nce3_loss,
                    neutral_penalty, quest_loss,
                    clip_ta, clip_tv, clip_av,
                    clip_mass_t, clip_mass_v, clip_mass_a, clip_mass_tav,
                    tfn_tv_loss, tfn_ta_loss, tfn_av_loss,
                    pred_val, truth_val,
                    pol_pred, pol_truth, polarity_correct,
                    is_neutral, neutral_ratio, warmup_status,
                    grad_norm, grad_clipped,
                    batch_size
                ]
                
                writer.writerow(row_data)
    
    def log_evaluation_epoch(self, epoch, phase, dataset_split, predictions, ground_truth, 
                           sample_ids, polarity_predictions=None, polarity_truth=None):
        """
        记录评估轮次的样本级别指标
        
        Args:
            epoch (int): 当前轮次
            phase (str): 阶段 ('valid', 'test')
            dataset_split (str): 数据集分割 ('dev', 'test')
            predictions (torch.Tensor): 预测值
            ground_truth (torch.Tensor): 真实值
            sample_ids (list): 样本ID列表
            polarity_predictions (torch.Tensor, optional): 极性预测
            polarity_truth (torch.Tensor, optional): 极性真值
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            preds = predictions.view(-1).cpu().detach().numpy()
        else:
            preds = np.array(predictions).reshape(-1)
            
        if isinstance(ground_truth, torch.Tensor):
            truths = ground_truth.view(-1).cpu().detach().numpy()
        else:
            truths = np.array(ground_truth).reshape(-1)
        
        # 处理极性预测和真值
        if polarity_predictions is not None and polarity_truth is not None:
            if isinstance(polarity_predictions, torch.Tensor):
                pol_preds = polarity_predictions.argmax(dim=-1).cpu().detach().numpy()
            else:
                pol_preds = np.array(polarity_predictions)
            
            if isinstance(polarity_truth, torch.Tensor):
                pol_truths = polarity_truth.cpu().detach().numpy()
            else:
                pol_truths = np.array(polarity_truth)
            
            # 确保数组是一维的
            if pol_preds.ndim == 0:
                pol_preds = np.array([pol_preds])
            if pol_truths.ndim == 0:
                pol_truths = np.array([pol_truths])
        else:
            pol_preds = np.zeros(len(preds), dtype=int)
            pol_truths = np.zeros(len(preds), dtype=int)
        
        # 为每个样本写入一行
        with open(self.eval_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for i in range(len(preds)):
                sample_id = sample_ids[i] if i < len(sample_ids) else f"eval_sample_{i}"
                
                # 样本级别的预测和真值
                pred_val = float(preds[i])
                truth_val = float(truths[i])
                
                # 安全地获取极性预测和真值
                if i < len(pol_preds):
                    pol_pred = int(pol_preds[i])
                else:
                    pol_pred = 0
                    
                if i < len(pol_truths):
                    pol_truth = int(pol_truths[i])
                else:
                    pol_truth = 0
                
                # 样本级别的误差
                absolute_error = abs(pred_val - truth_val)
                squared_error = (pred_val - truth_val) ** 2
                
                # 二分类标签（正负）
                binary_truth = 1 if truth_val >= 0 else 0
                binary_pred = 1 if pred_val >= 0 else 0
                binary_correct = (binary_pred == binary_truth)
                
                # 多级分类标签（7级：[-3,3]）
                mult_7_truth = int(np.clip(np.round(truth_val), -3, 3) + 3)  # 0-6
                mult_7_pred = int(np.clip(np.round(pred_val), -3, 3) + 3)    # 0-6
                mult_7_correct = (mult_7_pred == mult_7_truth)
                
                # 多级分类标签（5级：[-2,2]）
                mult_5_truth = int(np.clip(np.round(truth_val), -2, 2) + 2)  # 0-4
                mult_5_pred = int(np.clip(np.round(pred_val), -2, 2) + 2)    # 0-4
                mult_5_correct = (mult_5_pred == mult_5_truth)
                
                # 多级分类标签（3级：[-1,1]）
                mult_3_truth = int(np.clip(np.round(truth_val), -1, 1) + 1)  # 0-2
                mult_3_pred = int(np.clip(np.round(pred_val), -1, 1) + 1)    # 0-2
                mult_3_correct = (mult_3_pred == mult_3_truth)
                
                # 极性分类是否正确
                polarity_correct = (pol_pred == pol_truth)
                
                # 是否为非零样本
                is_non_zero = (truth_val != 0)
                
                # 构建样本级别的行数据
                row_data = [
                    timestamp, epoch, phase, dataset_split, i, sample_id,
                    pred_val, truth_val,
                    pol_pred, pol_truth, polarity_correct,
                    absolute_error, squared_error,
                    binary_truth, binary_pred, binary_correct,
                    mult_7_truth, mult_7_pred, mult_7_correct,
                    mult_5_truth, mult_5_pred, mult_5_correct,
                    mult_3_truth, mult_3_pred, mult_3_correct,
                    is_non_zero
                ]
                
                writer.writerow(row_data)
        
        # 计算并返回总体评估指标
        mae = np.mean(np.absolute(preds - truths))
        mse = np.mean((preds - truths) ** 2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(preds, truths)[0][1] if len(preds) > 1 else 0.0
        
        # 计算分类指标
        non_zeros = np.array([i for i, e in enumerate(truths) if e != 0])
        
        # F1分数和准确率
        binary_truth_has0 = truths >= 0
        binary_preds_has0 = preds >= 0
        f1_score_all = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')
        accuracy_all = accuracy_score(binary_truth_has0, binary_preds_has0)
        
        # 非零样本的F1和准确率
        if len(non_zeros) > 0:
            binary_truth_non0 = truths[non_zeros] > 0
            binary_preds_non0 = preds[non_zeros] > 0
            f1_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
            accuracy_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)
        else:
            f1_score_non0 = 0.0
            accuracy_non0 = 0.0
        
        # 多类准确率（不同范围）
        test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(truths, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(truths, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(truths, a_min=-1., a_max=1.)
        
        mult_acc_7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(truths))
        mult_acc_5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(truths))
        mult_acc_3 = np.sum(np.round(test_preds_a3) == np.round(test_truth_a3)) / float(len(truths))
        
        # 返回评估指标字典
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'correlation': correlation,
            'f1_score': f1_score_all,
            'f1_score_non0': f1_score_non0,
            'accuracy': accuracy_all,
            'accuracy_non0': accuracy_non0,
            'mult_acc_7': mult_acc_7,
            'mult_acc_5': mult_acc_5,
            'mult_acc_3': mult_acc_3
        }
    
    def log_sample_level_details(self, epoch, batch_idx, phase, sample_details):
        """
        记录样本级别的详细信息（可选，用于深度分析）
        
        Args:
            epoch (int): 当前轮次
            batch_idx (int): 批次索引  
            phase (str): 阶段
            sample_details (dict): 样本详细信息
        """
        # 这个方法可以用于记录更详细的样本级别信息
        # 比如每个样本的预测值、真值、损失等
        pass
    
    def close(self):
        """关闭日志记录器"""
        print(f"📊 日志记录器已关闭")
        print(f"   训练日志已保存到: {self.train_log_file}")
        print(f"   评估日志已保存到: {self.eval_log_file}")

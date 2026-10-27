import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from utils.training_logger import SampleLevelLogger
from model import MMIM
from modules.mpcf_modules import get_polarity_label


class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        # hyp_params==args
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        
        self.model = model
        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.nce2 = hp.nce2
        self.nce3 = hp.nce3
        self.y_true = torch.tensor([])
        self.y_pre = torch.tensor([])
        self.p_value =[]
        self.t_statistic = []

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = MMIM(hp)

        if torch.cuda.is_available():
            # if False:
            self.device = torch.device("cuda:0")
            # self.device = torch.device("cuda:7")
            model = model.to(self.device)

        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []
            
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)
                    
                for p in (mmilb_param+main_param):
                    if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        # nn.init.xavier_normal_()
                        nn.init.xavier_normal_(p)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )
        

        self.scheduler_main = ReduceLROnPlateau(
            self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)
        
        # 初始化日志记录器
        if self.is_train:
            self.logger = SampleLevelLogger(
                log_dir="../training_logs",  # 相对于src目录
                dataset_name=hp.dataset,
                experiment_name="MPCF"
            )

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        # 在方法开始处添加best_valid的初始化
        best_valid = float('inf')  # 初始化为无穷大
        best_mae = float('inf')  # 添加best_mae的初始化
        best_epoch = 0  # 添加best_epoch的初始化
        
        # 检查BERT参数是否已经损坏
        if self.check_and_reset_bert_if_corrupted():
            print("⚠️  BERT参数在训练开始前就已损坏，已重新初始化")
        
        model = self.model

        optimizer_main = self.optimizer_main

        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, optimizer, criterion,epochs, stage=1):
            epoch_loss = 0
            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            
            nce_loss = 0.0
            ba_loss = 0.0
    
            polarity_loss_total = 0.0
            polarity_acc_total = 0.0
            polarity_samples_total = 0
            
            # 📊 详细损失监控变量初始化
            mae_loss_batch = 0.0
            polarity_loss_batch = 0.0
    
            ncl_loss_batch = 0.0
            neutral_penalty_batch = 0.0
            
            start_time = time.time()

            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                model.zero_grad()
                dd = torch.device("cuda:0")
                # dd = torch.device("cuda:7")
                with torch.cuda.device(0):
                    # with torch.cuda.device(3):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.to(dd), visual.to(dd), audio.to(dd), y.to(dd), l.to(dd), bert_sent.to(dd), \
                        bert_sent_type.to(dd), bert_sent_mask.to(dd)

                    if self.hp.dataset == "ur_funny":
                        y = y.squeeze()

                batch_size = y.size(0)
                
                # 🔧 记录当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # visiual==(261，32，20) audio==(218,32,5)
                model_output = model(text, visual, audio, vlens, alens,
                                               bert_sent, bert_sent_type, bert_sent_mask, y)
                
                # 🚀 MGCF完全代替MCF：统一的输出格式处理（删除跨模态对比学习）
                # 模型始终返回MGCF格式：nce, preds, nce2, nce3, concepts, polarity_logits
                nce, preds, nce2, nce3, concepts, polarity_logits = model_output
                
                # QUEST相关变量设为0（已删除QUEST模块）
                quest_loss = torch.tensor(0.0, device=nce.device)
                
                # 检查模型输出是否包含nan/inf
                if torch.isnan(nce).any() or torch.isinf(nce).any():
                    print(f"❌ 警告: nce包含nan/inf，跳过此batch")
                    continue
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    print(f"❌ 警告: preds包含nan/inf，跳过此batch")
                    continue
                
                if stage == 1:
                    # 🎯 优化的损失权重分配策略 - 确保MAE主任务优先
                    # 🚀 MGCF完全代替MCF：统一的损失计算逻辑
                    # ================ 主任务：MAE（L1Loss）权重=1.0 ================
                    mae_loss = torch.abs(preds - y).mean()  # L1Loss (MAE) - 主要优化目标
                    
                    # ================ 辅助任务：轻权重引导 ================
                    # 极性分类监督损失（权重=0.1）
                    polarity_loss = torch.tensor(0.0, device=y.device)
                    if polarity_logits is not None:
                        polarity_label = get_polarity_label(y.view(-1))  # [B]
                        polarity_loss = F.cross_entropy(polarity_logits, polarity_label)
                    
                    # 🔥 第六步：Warmup策略 - 动态确定是否启用对比损失
                    enable_contrastive = True  # 默认启用对比损失
                    warmup_info = ""
                    
                    if hasattr(self.hp, 'enable_warmup') and self.hp.enable_warmup:
                        # 根据数据集选择Warmup轮数
                        if self.hp.dataset.lower() == 'mosi':
                            warmup_threshold = getattr(self.hp, 'warmup_epochs_mosi', 5)
                        elif self.hp.dataset.lower() == 'mosei':
                            warmup_threshold = getattr(self.hp, 'warmup_epochs_mosei', 6)
                        elif self.hp.dataset.lower() == 'sims':
                            warmup_threshold = getattr(self.hp, 'warmup_epochs_sims', 8)
                        else:
                            warmup_threshold = getattr(self.hp, 'warmup_epochs', 5)
                        
                        # 判断当前是否在Warmup阶段
                        if epochs < warmup_threshold:
                            enable_contrastive = False
                            warmup_info = f" [Warmup {epochs+1}/{warmup_threshold}]"
                        else:
                            warmup_info = f" [Post-Warmup]"
                    
                    # ================ 合理权重分配 ================
                    # 🔧 权重配置（可通过参数调整）
                    mae_weight = getattr(self.hp, 'mae_weight', 1.0)           # 主任务权重
                    polarity_weight = getattr(self.hp, 'polarity_weight', 0.1) # 极性分类权重
            
                    ncl_weight = getattr(self.hp, 'ncl_weight', 0.05)          # 原始MCL权重
                    
                    # 🚀 第五步：中性惩罚机制 - 惩罚模糊的0值预测
                    neutral_penalty = torch.tensor(0.0, device=y.device)
                    if hasattr(self.hp, 'enable_neutral_penalty') and self.hp.enable_neutral_penalty:
                        # 中性窗口：接近0的预测范围
                        neutral_threshold = getattr(self.hp, 'neutral_threshold', 0.25)
                        neutral_penalty_weight = getattr(self.hp, 'neutral_penalty_weight', 0.1)
                        neutral_penalty_strength = getattr(self.hp, 'neutral_penalty_strength', 0.5)
                        
                        # 计算中性惩罚：对接近0的预测加惩罚
                        is_neutral = (preds.abs() < neutral_threshold).float()  # [B, 1] 
                        neutral_penalty = (is_neutral * neutral_penalty_strength).mean()
                        
                        # 监控中性预测比例
                        neutral_ratio = is_neutral.mean().item()
                        if hasattr(self, '_neutral_ratios'):
                            self._neutral_ratios.append(neutral_ratio)
                        else:
                            self._neutral_ratios = [neutral_ratio]
                    
                    # 🎯 核心：MAE主导的损失组合 + 中性惩罚
                    y_loss = mae_weight * mae_loss + polarity_weight * polarity_loss + neutral_penalty
                    
                    # 📊 损失监控信息存储（用于日志打印）
                    current_mae_loss = mae_loss.item()
                    current_polarity_loss = polarity_loss.item()
                    current_quest_loss = quest_loss.item() if 'quest_loss' in locals() else 0.0
    
                    current_ncl_loss = (nce.item() + nce2.item() + nce3.item()) if 'nce' in locals() else 0.0
                    current_neutral_penalty = neutral_penalty.item() if 'neutral_penalty' in locals() else 0.0
                    
                    # 检查y_loss是否为nan/inf
                    if torch.isnan(y_loss).any() or torch.isinf(y_loss).any():
                        print(f"❌ 警告: y_loss包含nan/inf，跳过此batch")
                        continue

                    if self.hp.contrast:
                        # ================ 最终损失组合 ================
                        # 🚀 MGCF完全代替MCF：统一的损失权重分配策略 + 中性惩罚 + Warmup策略
                        neutral_penalty_weight = getattr(self.hp, 'neutral_penalty_weight', 0.1)
                        
                        # 🔥 Warmup策略：前N轮仅使用主任务+极性损失
                        if not enable_contrastive:
                            # Warmup阶段：仅主任务+极性分类
                            loss = (
                                mae_weight * mae_loss +                    # 主任务：MAE（权重=1.0）
                                polarity_weight * polarity_loss +          # 极性分类（权重=0.1）
                                neutral_penalty_weight * neutral_penalty   # 🚀 中性惩罚（权重=0.1）
                            )
                            # 对比损失在Warmup期间设为0，避免干扰主任务学习
                            current_quest_loss = 0.0
        
                            current_ncl_loss = 0.0
                        else:
                            # 正常阶段：全部损失启用
                            loss = (
                                mae_weight * mae_loss +                    # 主任务：MAE（权重=1.0）
                                polarity_weight * polarity_loss +          # 极性分类（权重=0.1）
                
                                ncl_weight * (nce + nce2 + nce3) +        # 原始MCL对比（权重=0.05）
                                neutral_penalty_weight * neutral_penalty   # 🚀 中性惩罚（权重=0.1）
                            )
                    else:
                        loss = y_loss
                    
                    # 检查总损失是否为nan/inf
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"❌ 警告: 总损失包含nan/inf，跳过此batch")
                        continue
                    
                    loss.backward()
                    
                    # ✅ 方案1：梯度裁剪 - 防止Cross-modal & InfoNCE损失导致梯度爆炸
                    gradient_clip_norm = getattr(self.hp, 'gradient_clip_norm', 5.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

                else:
                    raise ValueError('stage index can either be 0 or 1')

                # 📊 梯度范数监控（可选，用于调试）
                total_grad_norm = 0.0
                enable_gradient_monitoring = getattr(self.hp, 'enable_gradient_monitoring', False)
                if enable_gradient_monitoring:
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                            total_grad_norm = total_grad_norm ** 0.5
                
                    gradient_clip_norm = getattr(self.hp, 'gradient_clip_norm', 5.0)
                    # 仅在梯度范数超过裁剪阈值时显示
                    if total_grad_norm > gradient_clip_norm:
                        print(f"📊 梯度范数: {total_grad_norm:.2f} (已裁剪到{gradient_clip_norm})")
                
                # 📊 记录训练批次日志
                if hasattr(self, 'logger'):
                    # 准备损失字典
                    losses_dict = {
                        'total_loss': loss.item(),
                        'mae_loss': current_mae_loss,
                        'polarity_loss': current_polarity_loss,
                        'nce_loss': nce.item(),
                        'nce2_loss': nce2.item(), 
                        'nce3_loss': nce3.item(),
                        'neutral_penalty': current_neutral_penalty,
                        'quest_loss': current_quest_loss,
                        # 对比学习损失详情（需要从nce中分解）
                        'clip_ta': nce.item() / 3.0,  # 近似分解
                        'clip_tv': nce.item() / 3.0,
                        'clip_av': nce.item() / 3.0,
                        'clip_mass_t': nce2.item() / 4.0,  # 近似分解
                        'clip_mass_v': nce2.item() / 4.0,
                        'clip_mass_a': nce2.item() / 4.0,
                        'clip_mass_tav': nce2.item() / 4.0,
                        'tfn_tv_loss': nce3.item() / 3.0,  # 近似分解
                        'tfn_ta_loss': nce3.item() / 3.0,
                        'tfn_av_loss': nce3.item() / 3.0
                    }
                    
                    # 准备指标字典
                    metrics_dict = {
                        'lr': current_lr,
                        'polarity_acc': polarity_acc_current.item() if 'polarity_acc_current' in locals() else 0.0,
                        'polarity_samples': batch_size if polarity_logits is not None else 0,
                        'neutral_ratio': neutral_ratio if 'neutral_ratio' in locals() else 0.0,
                        'warmup_status': warmup_info if 'warmup_info' in locals() else '',
                        'grad_norm': total_grad_norm,
                        'grad_clipped': total_grad_norm > getattr(self.hp, 'gradient_clip_norm', 5.0)
                    }
                    
                    # 准备样本信息
                    sample_info_dict = {
                        'batch_size': batch_size,
                        'sample_ids': ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids
                    }
                    
                    # 确定训练阶段
                    phase_name = "warmup" if not enable_contrastive else "post_warmup"
                    
                    # 准备极性预测和真值
                    polarity_preds = None
                    polarity_truths = None
                    if polarity_logits is not None:
                        polarity_preds = polarity_logits
                        polarity_truths = get_polarity_label(y.view(-1))
                    
                    # 记录样本级别日志
                    self.logger.log_training_batch(
                        epoch=epochs,
                        batch_idx=i_batch,
                        phase=phase_name,
                        losses=losses_dict,
                        metrics=metrics_dict,
                        sample_info=sample_info_dict,
                        predictions=preds,
                        ground_truth=y,
                        polarity_predictions=polarity_preds,
                        polarity_truth=polarity_truths
                    )

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size
                ba_loss += nce2.item() * batch_size  # BA损失对应nce2（双模态对齐）

                
                # 📊 累积各组件损失（用于详细监控）- MGCF完全代替MCF后的统一监控
                if 'current_mae_loss' in locals():
                    mae_loss_batch += current_mae_loss * batch_size
                if 'current_polarity_loss' in locals():
                    polarity_loss_batch += current_polarity_loss * batch_size

                if 'current_ncl_loss' in locals():
                    ncl_loss_batch += current_ncl_loss * batch_size
                if 'current_neutral_penalty' in locals():
                    neutral_penalty_batch += current_neutral_penalty * batch_size
                
                # 统计极性分类损失和准确率
                if polarity_logits is not None:
                    polarity_label = get_polarity_label(y.view(-1))
                    polarity_loss_current = F.cross_entropy(polarity_logits, polarity_label)
                    pred_polarity = polarity_logits.argmax(dim=-1)
                    polarity_acc_current = (pred_polarity == polarity_label).float().mean()
                    
                    polarity_loss_total += polarity_loss_current.item() * batch_size
                    polarity_acc_total += polarity_acc_current.item() * batch_size
                    polarity_samples_total += batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size
                    avg_ba = ba_loss / proc_size
        
                    
                    # 计算极性分类准确率
                    avg_polarity_acc = polarity_acc_total / polarity_samples_total if polarity_samples_total > 0 else 0.0
                    avg_polarity_loss = polarity_loss_total / polarity_samples_total if polarity_samples_total > 0 else 0.0
                    
                    # 🚀 MGCF完全代替MCF后的统一日志输出
                    # 📊 计算各组件的平均损失
                    avg_mae_loss = mae_loss_batch / proc_size
                    avg_polarity_loss_component = polarity_loss_batch / proc_size

                    avg_ncl_loss_component = ncl_loss_batch / proc_size
                    avg_neutral_penalty_component = neutral_penalty_batch / proc_size
                    
                    # 🎯 详细损失监控打印 + 中性惩罚 + Warmup状态
                    neutral_info = f", NeutralP: {avg_neutral_penalty_component:.4f}" if hasattr(self.hp, 'enable_neutral_penalty') and self.hp.enable_neutral_penalty else ""
                    
                    # 🔥 Warmup状态显示
                    if hasattr(self.hp, 'enable_warmup') and self.hp.enable_warmup:
                        warmup_status = warmup_info
                    else:
                        warmup_status = ""
                    
                    print(f"[Epoch {epochs}]{warmup_status} Losses => MAE: {avg_mae_loss:.4f}, "
                          f"Polarity: {avg_polarity_loss_component:.4f}, "
  
                          f"NCEs: {avg_ncl_loss_component:.4f}{neutral_info}")
                    
                    # 常规训练日志
                    if polarity_logits is not None:
                                                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | NCE {:.3f} | BA {:.4f} | PolAcc {:.3f}'.
                        format(epochs, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                        avg_loss, avg_nce, avg_ba, avg_polarity_acc))
                    else:
                        print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                              format(epochs, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                                     avg_loss, avg_nce, avg_ba))
                    
                    # 重置累积变量
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
            
                    polarity_loss_total = 0.0
                    polarity_acc_total = 0.0
                    polarity_samples_total = 0
                    
                    # 📊 重置详细损失监控变量
                    mae_loss_batch = 0.0
                    polarity_loss_batch = 0.0
            
                    ncl_loss_batch = 0.0
                    neutral_penalty_batch = 0.0
                    
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, epochs,test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0

            results = []
            truths = []
            all_sample_ids = []
            all_polarity_preds = []
            all_polarity_truths = []
            
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                    dd = torch.device("cuda:0")
                    with torch.cuda.device(0):
                        text, audio, vision, y = text.to(
                            dd), audio.to(dd), vision.to(dd), y.to(dd)
                        lengths = lengths.to(dd)
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(dd
                                                                                 ), bert_sent_type.to(dd), bert_sent_mask.to(dd)
                        if self.hp.dataset == 'iemocap':
                            y = y.long()

                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0)

                    # we don't need lld and bound anymore
                    model_output = model(
                        text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)
                    
                    # 🚀 MGCF完全代替MCF：统一的输出格式处理
                    # 模型始终返回MGCF格式：nce, preds, nce2, nce3, concepts, polarity_logits
                    _, preds, _, _, _, polarity_logits_eval = model_output

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    loss = criterion(preds, y)
                    
                    # 检查评估阶段的nan
                    if torch.isnan(loss).any():
                        print(f"❌ 警告: 评估阶段发现nan损失")

                    total_loss += loss.item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
                    all_sample_ids.extend(ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids)
                    
                    # 收集极性预测和真值
                    if polarity_logits_eval is not None:
                        from modules.mpcf_modules import get_polarity_label
                        polarity_label = get_polarity_label(y.view(-1))
                        pred_polarity = polarity_logits_eval.argmax(dim=-1)
                        all_polarity_preds.extend(pred_polarity.cpu().tolist())
                        all_polarity_truths.extend(polarity_label.cpu().tolist())

            avg_loss = total_loss / \
                (self.hp.n_test if test else self.hp.n_valid)
 
            results = torch.cat(results)
            truths = torch.cat(truths)
            
            # 📊 记录评估日志
            if hasattr(self, 'logger'):
                dataset_split = "test" if test else "dev"
                phase = "test" if test else "valid"
                
                # 准备极性预测和真值
                polarity_preds_tensor = None
                polarity_truths_tensor = None
                if all_polarity_preds and all_polarity_truths and len(all_polarity_preds) > 0:
                    polarity_preds_tensor = torch.tensor(all_polarity_preds)
                    polarity_truths_tensor = torch.tensor(all_polarity_truths)
                
                eval_metrics = self.logger.log_evaluation_epoch(
                    epoch=epochs,
                    phase=phase,
                    dataset_split=dataset_split,
                    predictions=results,
                    ground_truth=truths,
                    sample_ids=all_sample_ids,
                    polarity_predictions=polarity_preds_tensor,
                    polarity_truth=polarity_truths_tensor
                )
            
            return avg_loss, results, truths
            
        patience = self.hp.patience
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            train_loss = train(model, optimizer_main, criterion, epoch,1)
            val_loss, _, _ = evaluate(model, criterion,epoch, test=False)
            test_loss, results, truths = evaluate(model, criterion,epoch, test=True)

            end = time.time()
            duration = end-start
            # Decay learning rate by validation loss
            scheduler_main.step(val_loss)

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                epoch, duration, val_loss, test_loss))
            print("-"*50)

            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'sims':
                        eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)

                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/best_model.pt!")
                    save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break
        
        torch.save(self.p_value,"mosi_mcl_mcf_p_value.pth")
        torch.save(self.t_statistic,"mosi_mcl_mcf_t_statistic.pth")

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'sims':
            self.best_dict = eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)
        sys.stdout.flush()
        
        # 关闭日志记录器
        if hasattr(self, 'logger'):
            self.logger.close()

    def check_and_reset_bert_if_corrupted(self):
        """检查BERT参数是否损坏，如果损坏则重新初始化"""
        bert_corrupted = False
        for name, param in self.model.named_parameters():
            if 'bert' in name and (torch.isnan(param).any() or torch.isinf(param).any()):
                print(f"❌ 发现损坏的BERT参数: {name}")
                bert_corrupted = True
                break
        
        if bert_corrupted:
            print("🔄 重新初始化BERT模型...")
            # 重新加载BERT模型
            from modules.encoders import LanguageEmbeddingLayer
            self.model.text_enc = LanguageEmbeddingLayer(self.hp)
            if torch.cuda.is_available():
                self.model.text_enc = self.model.text_enc.to(self.device)
            print("✅ BERT模型已重新初始化")
            return True
        return False



import os
import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI, SIMS, PAD, UNK

# 智能BERT模型选择：根据数据集自动选择合适的BERT模型
def get_bert_model_and_tokenizer(dataset_name=None):
    """根据数据集选择合适的BERT模型"""
    
    # 中文数据集使用中文BERT，英文数据集使用英文BERT
    if dataset_name and 'sims' in dataset_name.lower():
        # SIMS中文数据集：使用中文BERT
        model_name = 'bert-base-chinese'
        print(f"🈶 检测到中文数据集({dataset_name})，使用中文BERT: {model_name}")
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
            return model_name, tokenizer
        except Exception as e:
            print(f"⚠️ 无法下载中文BERT，回退到本地模型: {e}")
            # 回退到本地模型
            # local_path = os.path.join(os.path.dirname(__file__), 'bert')
            local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bert'))
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=local_path, do_lower_case=True)
            return local_path, tokenizer
    else:
        # MOSI/MOSEI英文数据集：使用本地英文BERT
        # local_path = os.path.join(os.path.dirname(__file__), 'bert')
        local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bert'))
        print(f"🔤 检测到英文数据集({dataset_name})，使用本地英文BERT: {local_path}")
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=local_path, do_lower_case=True)
        return local_path, tokenizer

# 全局变量，将在MSADataset中根据数据集类型动态设置
bert_model_path = None
bert_tokenizer = None



class MSADataset(Dataset):
    def __init__(self, config):
        global bert_model_path, bert_tokenizer
        
        self.config = config

        # 检测数据集类型并设置相应的BERT模型
        dataset_name = None
        if "mosi" in str(config.data_dir).lower():
            dataset_name = "mosi"
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset_name = "mosei"
            dataset = MOSEI(config)
        elif "sims" in str(config.data_dir).lower():
            dataset_name = "sims"
            dataset = SIMS(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        # 根据数据集类型选择BERT模型
        bert_model_path, bert_tokenizer = get_bert_model_and_tokenizer(dataset_name)
        
        # 返回对齐的数据
        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.word2id = self.word2id
        config.bert_model_path = bert_model_path  # 传递给配置，供模型使用
        # config.pretrained_emb = self.pretrained_emb

    @property 
    def tva_dim(self):
        t_dim = 768

        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)
    
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

   

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''  

        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            if len(sample[0]) > 4:  # unaligned case 
                v_lens.append(torch.IntTensor([sample[0][4]]))  
                a_lens.append(torch.IntTensor([sample[0][5]]))  
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))  
            ids.append(sample[2]) 
       
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)  
        labels = torch.cat(labels, dim=0)  # torch.Size([32, 1]) 

        # Handle different label formats for different datasets
        if len(labels.shape) > 1 and labels.size(1) == 7:
        # MOSEI sentiment labels locate in the first column of sentiment matrix
            labels = labels[:, 0][:, None]
        elif len(labels.shape) == 1:
            # SIMS labels are single dimension, reshape to (batch_size, 1)
            labels = labels[:, None]

        # Rewrite this

        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:  
                out_dims = (max_len, len(sequences)) + trailing_dims
            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:  
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor
        a = [torch.LongTensor(sample[0][0])for sample in batch]
        b = [torch.FloatTensor(sample[0][1])for sample in batch]
        c = [torch.FloatTensor(sample[0][2])for sample in batch]
        sentences = pad_sequence([torch.LongTensor(sample[0][0])
                                  for sample in batch], padding_value=PAD)
        
        visual = pad_sequence([torch.FloatTensor(sample[0][1])
                               for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2])
                                 for sample in batch], target_len=alens.max().item())

        # BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = 50
        # Create bert indices using tokenizer
        
        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first 
        bert_sentences = torch.LongTensor(
            [sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor(
            [sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor(
            [sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids
   
#    # 根据CUDA可用性设置generator设备
#     if torch.cuda.is_available():
#         generator = torch.Generator(device='cuda')
#     else:
#         generator = torch.Generator(device='cpu')
    # DataLoader的generator必须在CPU上
    # 即使使用CUDA训练，generator也必须是CPU设备
    generator = torch.Generator(device='cpu')
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,   
        collate_fn=collate_fn,   
        generator=generator)  

    return data_loader

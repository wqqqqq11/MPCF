import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling


def return_unk():
    return UNK


def get_length(x):

    return x.shape[1]-(np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        # CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'
        CACHE_PATH = os.path.join(DATA_PATH, 'embedding_and_mapping.pt')

        # If cached data if already exists
        try:
            # self.train = load_pickle(DATA_PATH + '/train.pkl')
            # self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            # self.test = load_pickle(DATA_PATH + '/test.pkl')
            # self.pretrained_emb, self.word2id = None, None
            self.train = load_pickle(os.path.join(DATA_PATH, 'train.pkl'))
            self.dev = load_pickle(os.path.join(DATA_PATH, 'dev.pkl'))
            self.test = load_pickle(os.path.join(DATA_PATH, 'test.pkl'))
            self.pretrained_emb, self.word2id = None, None

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # load pickle file for unaligned acoustic and visual source
            # pickle_filename = '../datasets/MOSI/mosi_data_noalign.pkl'
            # csv_filename = '../datasets/MOSI/MOSI-label.csv'

            # pickle_filename = 'D:\PythonWorkplace\MCL-MCF-main\MCL-MCF\datasets\MOSI\mosi_data_noalign.pkl'
            # csv_filename = 'D:\PythonWorkplace\MCL-MCF-main\MCL-MCF\datasets\MOSI\MOSI-label.csv'

            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'MOSI'))
            pickle_filename = os.path.join(base_dir, 'mosi_data_noalign.pkl')
            csv_filename = os.path.join(base_dir, 'MOSI-label.csv')
            
            with open(pickle_filename, 'rb') as f:
 
                d = pickle.load(f)

            # read csv file for label and text   004 :1 cid_id
            # vid 000'03bSnISJMiM'001:'03bSnISJMiM'002:'03bSnISJMiM'003:'03bSnISJMiM'004:'03bSnISJMiM'
            # text 001:'THERE IS SAD PART'
            df = pd.read_csv(csv_filename)
            text = df['text']  
            vid = df['video_id']  
            cid = df['clip_id']  
            train_split_noalign = d['train']  # 1284
            dev_split_noalign = d['valid']  # 229
            test_split_noalign = d['test']  # 686

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)_(.*)')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            if True:
                v = np.concatenate(
                    (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
                vlens = get_length(v)

                a = np.concatenate(
                    (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
                alens = get_length(a)
                
                label = np.concatenate(
                    (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
                # label[label > 0] = 2
                # label[label < 0] = -2
                print(label.sort())
                L_V = v.shape[1]
                L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            
            all_id_list = list(
                map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset
                
                idd1, idd2 = re.search(pattern, idd).group(1, 2)

                # matching process
                try:
                    index = all_csv_id.index((idd1, idd2))
                except:
                    exit()
                """
                    Retrive noalign data from pickle file 
                """
                
                _words = text[index].split()
                _label = label[i].astype(np.float32)  
                _visual = v[i]  
                _acoustic = a[i]  
                _vlen = vlens[i]  
                _alen = alens[i]  
                _id = all_id[i]

                # remove nan values 
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                # For non-align setting
                # we also need to record sequence lengths
                """TODO: Add length counting for other datasets 
                """
                for word in _words:
                    actual_words.append(word)
                    words.append(word2id[word])

                
                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                # z-normalization per instance and remove nan/infs
                # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                
                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):
        if mode == "train":
            
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/trainsss.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'MOSEI'))
            pickle_filename = os.path.join(base_dir, 'mosei_senti_data_noalign.pkl')
            csv_filename = os.path.join(base_dir, 'MOSEI-label.csv')

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # pattern = re.compile('(.*)\[.*\]')
            pattern = re.compile('(.*)_([.*])')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            v = np.concatenate(
                (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)

            a = np.concatenate(
                (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
            alens = get_length(a)

            label = np.concatenate(
                (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset

                # matching process
                try:
                    index = i
                except:
                    import ipdb
                    ipdb.set_trace()

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = '{}[{}]'.format(all_csv_id[0], all_csv_id[1])

                # remove nan values
                # label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                for word in _words:
                    actual_words.append(word)
                    words.append(word2id[word])

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            # print(f"Total number of {num_drop} datapoints have been dropped.")
            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)
            self.pretrained_emb = None

            
            to_pickle(train, DATA_PATH + '/trainsss.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class SIMS:
    def __init__(self, config):
        DATA_PATH = str(config.dataset_dir)
        
        # If cached data already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None
        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # Load SIMS data files
            pickle_filename = os.path.join(DATA_PATH, 'sims_data_noalign.pkl')
            csv_filename = os.path.join(DATA_PATH, 'SIMS-label.csv')

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # Read CSV file for labels and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id'] 
            cid = df['clip_id']
            
            # Build CSV ID tuples - keep clip_id as integer for matching
            all_csv_id = [(vid[i], cid[i]) for i in range(len(vid))]

            train_split = d['train']
            dev_split = d['valid'] 
            test_split = d['test']

            # Sentinel epsilon for safe division
            EPS = 1e-6

            # Create word2id for SIMS (using simple integer mapping for now)
            from collections import defaultdict
            word2id = defaultdict(lambda: len(word2id))
            word2id['<pad>'] = 0
            word2id['<unk>'] = 1
            
            # Place holders for final datasets
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            num_drop = 0

            # Process each split
            for split_name, split_data in [('train', train_split), ('valid', dev_split), ('test', test_split)]:
                for i in range(len(split_data['id'])):
                    sample_id = split_data['id'][i]
                    
                    # Get features from pkl
                    if 'text_bert' in split_data:
                        # Use BERT features if available
                        _text_feat = split_data['text_bert'][i]
                    else:
                        _text_feat = split_data['text'][i]
                    
                    _visual = split_data['vision'][i]
                    _acoustic = split_data['audio'][i] 
                    _vlen = split_data['vision_lengths'][i] if 'vision_lengths' in split_data else len(_visual)
                    _alen = split_data['audio_lengths'][i] if 'audio_lengths' in split_data else len(_acoustic)
                    
                    # Parse pkl ID like MOSI does
                    # pkl: video_0020$_$0013 -> video_id='video_0020', clip_id='0013'
                    if '$_$' in sample_id:
                        parts = sample_id.split('$_$')
                        if len(parts) == 2:
                            idd1, idd2 = parts[0], parts[1]
                        else:
                            print(f"Warning: Unexpected ID format {sample_id}, skipping...")
                            num_drop += 1
                            continue
                    else:
                        print(f"Warning: ID {sample_id} doesn't contain $_ $, skipping...")
                        num_drop += 1
                        continue
                    
                    # Matching process: find by video_id and clip_id separately
                    # Convert clip_id string to int for matching (PKL has "0008", CSV has 8)
                    try:
                        # Find index where video_id matches idd1 and clip_id matches int(idd2)
                        idd2_int = int(idd2)  # Convert "0008" to 8
                        index = None
                        for idx, (csv_vid, csv_cid) in enumerate(all_csv_id):
                            if csv_vid == idd1 and csv_cid == idd2_int:
                                index = idx
                                break
                        
                        if index is None:
                            raise ValueError(f"No match found")
                    except (ValueError, TypeError) as e:
                        print(f"Warning: ID {sample_id} -> ({idd1}, {idd2}) not found in CSV, skipping...")
                        num_drop += 1
                        continue
                    
                    # Get label and text from CSV using found index
                    # 对中文文本进行正确的分词处理
                    raw_text = text[index]
                    if any('\u4e00' <= char <= '\u9fff' for char in raw_text):
                        # 包含中文字符，使用字符级分词（简单有效）
                        _words = list(raw_text.replace(' ', ''))  # 移除空格后按字符分词
                    else:
                        # 英文文本，使用空格分词
                        _words = raw_text.split()
                    _label = np.array([df.iloc[index]['label']], dtype=np.float32)

                    # Remove NaN values
                    _visual = np.nan_to_num(_visual)
                    _acoustic = np.nan_to_num(_acoustic)

                    # Process words for compatibility
                    actual_words = []
                    words = []
                    for word in _words:
                        actual_words.append(word)
                        words.append(word2id[word])

                    # Trim sequences to actual lengths
                    visual = _visual[-_vlen:, :] if _vlen < len(_visual) else _visual
                    acoustic = _acoustic[-_alen:, :] if _alen < len(_acoustic) else _acoustic

                    # Add to appropriate split
                    sample = ((words, visual, acoustic, actual_words, _vlen, _alen), _label, sample_id)
                    
                    if split_name == 'train':
                        train.append(sample)
                    elif split_name == 'valid':
                        dev.append(sample)
                    elif split_name == 'test':
                        test.append(sample)

            print(f"SIMS dataset processed. Dropped {num_drop} samples.")
            print("Dataset split:")
            print(f"Train Set: {len(train)}")
            print(f"Validation Set: {len(dev)}")
            print(f"Test Set: {len(test)}")
            
            word2id.default_factory = return_unk

            # Save processed data
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

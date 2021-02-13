import os
import torch
import random
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling

from m2p_mask import process_train_MAM_data

HALF_BATCHSIZE_TIME = 2000

class BaseDataset(Dataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False):

        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        # drop the results with some blank txt
        self.table = self.table.dropna(axis=0)

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
    
    def __len__(self):
        return len(self.X_a)


def load_acoustic_data(npy_path, npy_root=None):
    return torch.FloatTensor(np.load(os.path.join(npy_root, npy_path)))

def load_semantic_data(txt_path, txt_root=None, tokenizer=None):
    txt_content = ' '.join([x.strip('\n').split(',')[2].lower() for x in open(os.path.join(txt_root, txt_path),'r').readlines()])
    txt_content = tokenizer(txt_content)
    return txt_content

class MultiModalDataset(BaseDataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False, 
                 acoustic_config=None, semantic_config=None, tokenizer=None):
        super().__init__(file_path, sets, bucket_size, max_timestep, drop)
        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.mlm_collater = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        self.sample_step = 0

        X_a = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        X_t = self.table['align_path'].tolist()
        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_len, batch_x_t = [], [], []

        for x_a, x_len, x_t in zip(X_a, X_lens, X_t):
            batch_x_a.append(x_a)
            batch_len.append(x_len)
            batch_x_t.append(x_t)
            
            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size//2])
                    self.X_a.append(batch_x_a[bucket_size//2:])
                    self.X_t.append(batch_x_t[:bucket_size//2])
                    self.X_t.append(batch_x_t[bucket_size//2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_len, batch_x_t = [], [], []

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)
        
        assert len(self.X_a) == len(self.X_t)

    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,),config=self.acoustic_config
        )
        # preprocess with the semantic inputs
        x_t_pad_batch = self.tokenizer.pad(x_t_pad_batch, return_tensors="pt")
        s_inputs, s_labels = self.mlm_collater.mask_tokens(x_t_pad_batch['input_ids'])
        s_attention_mask = x_t_pad_batch['attention_mask']
        s_valid_batchid = torch.nonzero(torch.sum(s_labels!=-100,dim=1),as_tuple=False).view(-1)
        #---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        s_valid = torch.zeros(s_labels.size(0))
        s_valid[s_valid_batchid] = 1
        valid_batchid = a_valid.long() & s_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        #---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        #---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        #---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (s_inputs,s_attention_mask,s_labels)

    def __getitem__(self, index):
        acoustic_batch = [load_acoustic_data(x_file, self.root) for x_file in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)
        semantic_batch = [load_semantic_data(x_file, self.root, self.tokenizer) for x_file in self.X_t[index]]
        x_t_pad_batch = dict()
        x_t_pad_batch['input_ids'] = [x['input_ids'] for x in semantic_batch]
        x_t_pad_batch['attention_mask'] = [x['attention_mask'] for x in semantic_batch]
        return self.process_x_pad_batch(x_a_pad_batch, x_t_pad_batch)


class MultiModalDataset_Chinese(BaseDataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False, 
                 acoustic_config=None, semantic_config=None, tokenizer=None):
        super().__init__(file_path, sets, bucket_size, max_timestep, drop)
        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.mlm_collater = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        self.sample_step = 0

        X_a = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        X_t = self.table['transcript'].tolist()
        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_len, batch_x_t = [], [], []

        for x_a, x_len, x_t in zip(X_a, X_lens, X_t):
            batch_x_a.append(x_a)
            batch_len.append(x_len)
            batch_x_t.append(x_t)
            
            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size//2])
                    self.X_a.append(batch_x_a[bucket_size//2:])
                    self.X_t.append(batch_x_t[:bucket_size//2])
                    self.X_t.append(batch_x_t[bucket_size//2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_len, batch_x_t = [], [], []

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)
        
        assert len(self.X_a) == len(self.X_t)

    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,),config=self.acoustic_config
        )
        # preprocess with the semantic inputs
        x_t_pad_batch = self.tokenizer.pad(x_t_pad_batch, return_tensors="pt")
        s_inputs, s_labels = self.mlm_collater.mask_tokens(x_t_pad_batch['input_ids'])
        s_attention_mask = x_t_pad_batch['attention_mask']
        s_valid_batchid = torch.nonzero(torch.sum(s_labels!=-100,dim=1),as_tuple=False).view(-1)
        #---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        s_valid = torch.zeros(s_labels.size(0))
        s_valid[s_valid_batchid] = 1
        valid_batchid = a_valid.long() & s_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        #---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        #---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        #---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (s_inputs,s_attention_mask,s_labels)

    def __getitem__(self, index):
        acoustic_batch = [load_acoustic_data(x_file, self.root) for x_file in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)
        semantic_batch = [self.tokenizer(x) for x in self.X_t[index]]
        x_t_pad_batch = dict()
        x_t_pad_batch['input_ids'] = [x['input_ids'] for x in semantic_batch]
        x_t_pad_batch['attention_mask'] = [x['attention_mask'] for x in semantic_batch]
        return self.process_x_pad_batch(x_a_pad_batch, x_t_pad_batch)
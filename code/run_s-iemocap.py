import os
import re
import time
import pickle
import numpy as np
from functools import reduce

import torch
import torch.nn as nn

from transformers import AdamW, RobertaTokenizer
from transformers import RobertaForSequenceClassification

class EmotionDataset(object):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        audio_name = re.sub('\.npy','.wav',os.path.basename(audio_path))
        text_words = [x.lower() for x in re.split(' +',re.sub('[\.,\?\!]',' ', asr_text))]
        text_input = self.tokenizer(' '.join(text_words))
        return {'text_input':text_input,'label':label,'audio_name':audio_name}
    
def collate(sample_list, tokenizer):
    pad_batch_text = {
        'input_ids':[x['text_input']['input_ids'] for x in sample_list],
        'attention_mask':[x['text_input']['attention_mask'] for x in sample_list],
    }
    pad_batch_text = tokenizer.pad(pad_batch_text, return_tensors='pt')
    s_inputs = pad_batch_text['input_ids']
    s_attention_mask = pad_batch_text['attention_mask']
    
    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((s_inputs, s_attention_mask), batch_label, batch_name)


def wa_helper(pred_label, true_label):
    result = np.mean(pred_label == true_label)
    return result


def ua_helper(pred_label, true_label, class_num=4):
    pred_onehot = np.eye(class_num)[pred_label]
    true_onehot = np.eye(class_num)[true_label]
    result = np.mean(np.sum((pred_onehot == true_onehot)*true_onehot,axis=0) / np.sum(true_onehot,axis=0))
    return result

if __name__ == '__main__':

    data_source = ['/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session1.pkl',
                   '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session2.pkl',
                   '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session3.pkl',
                   '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session4.pkl',
                   '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session5.pkl']

    data_root = '/dataset/iemocap_mel160/V1_0/iemocap_mel160/'

    best_results = []
    epoch_num = 20

    for fold_num in range(5):
        print('Start to work on fold {}'.format(fold_num))
        
        # Define model
        model = RobertaForSequenceClassification.from_pretrained(
            '/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-tiny/', num_labels=4)
        model.cuda()
        
        # Define tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(
            "../tokenizer/libri-roberta_train-960/")
        
        # Define optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        
        # Define dataloader
        valid_data = pickle.load(open(data_source[fold_num],'rb'))
        valid_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in valid_data]

        train_data = data_source[:fold_num] + data_source[fold_num+1:]
        train_data = list(reduce(lambda a,b: a+b, [pickle.load(open(x, 'rb')) for x in train_data]))
        train_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in train_data]

        batch_size = 16
        num_workers = 4
        
        train_dataset = EmotionDataset(train_data, tokenizer)
        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x, tokenizer),
            shuffle = True, num_workers = num_workers
        )
        valid_dataset = EmotionDataset(valid_data, tokenizer)
        valid_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x, tokenizer),
            shuffle = False, num_workers = num_workers
        )
        
        # Start to Train
        best_wa, save_ua = 0, 0
        for epoch in range(epoch_num):
            epoch_train_loss = []
            model.train()
            start_time = time.time()
                
            for semantic_inputs, label_inputs, _ in train_loader:
                input_ids = semantic_inputs[0].cuda()
                attention_mask = semantic_inputs[1].cuda()
                labels = label_inputs.cuda()
                
                model.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                epoch_train_loss.append(loss)
                loss.backward()
                optimizer.step()

            epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

            pred_y, true_y = [], []
            model.eval()

            for semantic_inputs, label_inputs, _ in valid_loader:
                input_ids = semantic_inputs[0].cuda()
                attention_mask = semantic_inputs[1].cuda()
                true_y.extend(label_inputs.numpy())
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=None)
                label_outputs = torch.argmax(outputs.logits,axis=1).cpu().detach().numpy().astype(int)
                pred_y.extend(list(label_outputs))

            pred_y = np.array(pred_y)
            true_y = np.array(true_y)

            wa = wa_helper(pred_y, true_y)
            ua = ua_helper(pred_y, true_y)

            elapsed_time = time.time() - start_time
            print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            print("WA: {:.3f} - UA: {:.3f} - Train Loss: {:.3f}".format(wa, ua, epoch_train_loss))
            
            if wa > best_wa or (ua > save_ua and (ua-save_ua)/(best_wa-wa)>=2):
                best_wa, save_ua = wa, ua
        
        best_results.append({'fold_num':fold_num,'best_wa':best_wa, 'best_ua':save_ua})

        print('Work finished on fold {}'.format(fold_num))

    pickle.dump(best_results, open('/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-tiny/iemocap_report.pkl','wb'))
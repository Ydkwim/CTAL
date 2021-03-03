import os
import re
import time
import yaml
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import RobertaTokenizer
from transformers import AdamW

from m2p_mask import process_test_MAM_data
from m2p_transfer import RobertaM2Downstream

from calculate_eer import get_eer

# Here we need to recalculate the metrics for different tasks

def downstream_metrics(pred_label, true_label, task='emotion'):
    """Calculate the evluation metrics for downstream tasks
    pred_label: prediction results
    true_label: human annotations
    task: the name of the downstream task
    """
    assert task in ['emotion','sentiment','verification']
    pred_label, true_label = np.array(pred_label), np.array(true_label)
    
    if task == 'emotion':
        wa = np.mean(pred_label.astype(int) == true_label.astype(int))
        pred_onehot = np.eye(4)[pred_label.astype(int)]
        true_onehot = np.eye(4)[true_label.astype(int)]
        ua = np.mean(np.sum((pred_onehot==true_onehot)*true_onehot,axis=0)/np.sum(true_onehot,axis=0))
        key_metric, report_metric = 0.9*wa+0.1*ua, {'wa':wa,'ua':ua}
    
    elif task == 'sentiment':
        mae = np.mean(np.abs(pred_label - true_label))
        corr = np.corrcoef(pred_label, true_label)[0][1]
        nonzero_index = np.where(true_label!=0)
        pred_2class = (pred_label[nonzero_index] > 0).astype(int)
        true_2class = (true_label[nonzero_index] > 0).astype(int)
        acc_2class = accuracy_score(true_2class,pred_2class)
        f1_2class = f1_score(true_2class, pred_2class, average='weighted')
        key_metric, report_metric = -1.0 * mae, {'mae':mae,'corr':corr,'acc_2class':acc_2class,'f1_2class':f1_2class}

    else:
        eer = get_eer(pred_label, true_label)
        key_metric, report_metric = -1.0 * eer, {'eer':eer}

    return key_metric, report_metric


class DownstreamDataset(object):
    def __init__(self, data_list, tokenizer, audio_length=None):
        self.data_list = data_list
        self.audio_length = audio_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        audio_name = re.sub('\.npy','.wav',os.path.basename(audio_path))
        audio_input = torch.FloatTensor(np.load(audio_path))
        if self.audio_length is not None: audio_input=audio_input[:self.audio_length,:]
        # Here sometimes the asr input could be the sepearte file path
        if os.path.isfile(asr_text):
            # The following preprocess could be modified based on the formats of the text record
            asr_text = ' '.join([x.strip('\n').split(',')[2] for x in open(asr_text,'r').readlines()])
        text_words = [x.lower() for x in re.split(' +',re.sub('[\.,\?\!]',' ', asr_text))]
        text_input = self.tokenizer(' '.join(text_words))
        return {'audio_input':audio_input,'text_input':text_input,'label':label,'audio_name':audio_name}


def collate(sample_list, tokenizer, config):
    batch_audio = [x['audio_input'] for x in sample_list]
    pad_batch_audio = pad_sequence(batch_audio, batch_first=True)
    
    pad_batch_text = {
        'input_ids':[x['text_input']['input_ids'] for x in sample_list],
        'attention_mask':[x['text_input']['attention_mask'] for x in sample_list],
    }
    pad_batch_text = tokenizer.pad(pad_batch_text, return_tensors='pt')
    s_inputs = pad_batch_text['input_ids']
    s_attention_mask = pad_batch_text['attention_mask']
    
    a_attention_mask, a_inputs = process_test_MAM_data((pad_batch_audio,),config)

    batch_label = torch.tensor([x['label'] for x in sample_list])
    batch_name = [x['audio_name'] for x in sample_list]

    return ((a_inputs, a_attention_mask), 
            (s_inputs, s_attention_mask),
            batch_label, batch_name)


def run(args, config, train_data, valid_data, test_data=None):
    ############################ PARAMETER SETTING ##########################
    num_workers = config['dataloader']['n_jobs']
    batch_size = config['dataloader']['batch_size']
    # learning_rate = config['optimizer']['learning_rate']
    # warmup_proportion = config['optimizer']['warmup_proportion']
    # save_ckpt_dir = os.path.join(args.save_path, 'checkpoints')

    audio_length = 3000
    epochs = args.epochs

    tokenizer = RobertaTokenizer.from_pretrained(config['upstream']['semantic']['tokenizer_path'])
    ############################## PREPARE DATASET ##########################
    train_dataset = DownstreamDataset(train_data, tokenizer, audio_length)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x,tokenizer,config['upstream']['acoustic']),
        shuffle = True, num_workers = num_workers
    )
    valid_dataset = DownstreamDataset(valid_data, tokenizer, audio_length)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x,tokenizer,config['upstream']['acoustic']),
        shuffle = False, num_workers = num_workers
    )
    
    if test_data is None:
        test_data = valid_data
    test_dataset = DownstreamDataset(test_data, tokenizer, audio_length)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x,tokenizer,config['upstream']['acoustic']),
        shuffle = False, num_workers = num_workers
    )
    ########################### CREATE MODEL #################################
    model = RobertaM2Downstream(config['upstream']['ckpt_path'],config['downstream']['label_num'],
                                freeze=config['upstream']['freeze'],
                                orthogonal_fusion=config['downstream']['orthogonal_fusion'])
    model.cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ########################### TRAINING #####################################
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

    for epoch in range(epochs):
        epoch_train_loss = []
        model.train()
        start_time = time.time()

        time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for acoustic_inputs, semantic_inputs, label_inputs, _ in progress:
            a_inputs = acoustic_inputs[0].cuda()
            a_attention_mask = acoustic_inputs[1].cuda()
            s_inputs = semantic_inputs[0].cuda()
            s_attention_mask = semantic_inputs[1].cuda()
            
            label_inputs = label_inputs.cuda()
            
            model.zero_grad()
            _, logits, loss = model(
                s_inputs=s_inputs,
                s_attention_mask=s_attention_mask,
                a_inputs=a_inputs,
                a_attention_mask=a_attention_mask,
                labels=label_inputs,
            )

            epoch_train_loss.append(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            count += 1
            
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, acc_train_loss))

        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
            for acoustic_inputs, semantic_inputs, label_inputs, _ in valid_loader:
                a_inputs = acoustic_inputs[0].cuda()
                a_attention_mask = acoustic_inputs[1].cuda()
                s_inputs = semantic_inputs[0].cuda()
                s_attention_mask = semantic_inputs[1].cuda()

                true_y.extend(list(label_inputs.numpy()))

                hiddens, logits, _ = model(
                    s_inputs=s_inputs,
                    s_attention_mask=s_attention_mask,
                    a_inputs=a_inputs,
                    a_attention_mask=a_attention_mask,
                    labels=None
                )

                if model.label_num == 1:
                    prediction = logits.view(-1)
                    label_outputs = prediction.cpu().detach().numpy().astype(float)
                else:
                    if args.task_name == "verification":
                        # for speaker verification we take the hidden before the classifier as the output
                        label_outputs = hiddens.cpu().detach().numpy().astype(float)
                    else:
                        prediction = torch.argmax(logits, axis=1)
                        label_outputs = prediction.cpu().detach().numpy().astype(int)

                pred_y.extend(list(label_outputs))

        # think about the metric calculation
        key_metric, report_metric = downstream_metrics(pred_y, true_y, args.task_name)
        
        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
            epoch_train_loss))

        if key_metric > best_metric:
            best_metric, best_epoch = key_metric, epoch
            print('Better Metric found on dev, calculate performance on Test')
            pred_y, true_y = [], []
            with torch.no_grad():
                time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
                for acoustic_inputs, semantic_inputs, label_inputs, _ in test_loader:
                    a_inputs = acoustic_inputs[0].cuda()
                    a_attention_mask = acoustic_inputs[1].cuda()
                    s_inputs = semantic_inputs[0].cuda()
                    s_attention_mask = semantic_inputs[1].cuda()

                    true_y.extend(list(label_inputs.numpy()))
                    
                    hiddens, logits, _, = model(
                        s_inputs=s_inputs,
                        s_attention_mask=s_attention_mask,
                        a_inputs=a_inputs,
                        a_attention_mask=a_attention_mask,
                        labels=None
                    )
                    
                    if model.label_num == 1:
                        prediction = logits.view(-1)
                        label_outputs = prediction.cpu().detach().numpy().astype(float)
                    else:
                        if args.task_name == "verification":
                            label_outputs = hiddens.cpu().detach().numpy().astype(float)
                        else:
                            prediction = torch.argmax(logits, axis=1)
                            label_outputs = prediction.cpu().detach().numpy().astype(int)

                    pred_y.extend(list(label_outputs))

            _, save_metric = downstream_metrics(pred_y, true_y, args.task_name)
            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))
            
    print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
    return save_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default=None, help='downstream task name')
    parser.add_argument("--config", type=str, default=None, help='configuration file path')
    parser.add_argument("--epochs", type=int, default=20, help="training epoches")
    parser.add_argument("--save_path", type=str, default=None, help="report or ckpt save path")
    parser.add_argument("--freeze", type=bool, default=False, help="freeze the pretrain model")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    report_result = []

    if args.task_name == 'emotion':
        
        data_root = '/dataset/iemocap_mel160/V1_0/iemocap_mel160/'
        data_source = ['/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session1.pkl',
                       '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session2.pkl',
                       '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session3.pkl',
                       '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session4.pkl',
                       '/dataset/iemocap_mel160/V1_0/iemocap_mel160/Session5.pkl']
        
        for i in range(5):
            valid_name = os.path.basename(data_source[i]).split('.pkl')[0]
            valid_data = pickle.load(open(data_source[i],'rb'))
            valid_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in valid_data]

            train_data = data_source[:i] + data_source[i+1:]
            train_data = list(reduce(lambda a,b: a+b, [pickle.load(open(x, 'rb')) for x in train_data]))
            train_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in train_data]

            report_metric = run(args, config, train_data, valid_data, test_data=None)
            report_result.append(report_metric)

    elif args.task_name == 'sentiment':
        data_root = '/dataset/mosei_mel160/V1_0/mel160/mel_160/'
        
        train_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/train.pkl','rb'))
        train_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in train_data]
            
        valid_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/dev.pkl','rb'))
        valid_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in valid_data]

        test_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/test.pkl','rb'))
        test_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in test_data]

        report_metric = run(args, config, train_data, valid_data, test_data)
        report_result = [report_metric]

    else:
        data_root = '/dataset/libri_mel160/V2_0/libri_mel160'

        train_data = pickle.load(open('/dataset/libri_mel160/V2_0/libri_mel160/verify_train.pkl','rb'))
        train_data = [(os.path.join(data_root,x[0]),os.path.join(data_root,x[1]),x[2]) for x in train_data]

        valid_data = pickle.load(open('/dataset/libri_mel160/V2_0/libri_mel160/verify_dev.pkl','rb'))
        valid_data = [(os.path.join(data_root,x[0]),os.path.join(data_root,x[1]),x[2]) for x in valid_data]

        test_data = pickle.load(open('/dataset/libri_mel160/V2_0/libri_mel160/verify_test.pkl','rb'))
        test_data = [(os.path.join(data_root,x[0]),os.path.join(data_root,x[1]),x[2]) for x in test_data]

        report_metric = run(args, config, train_data, valid_data, test_data)
        report_result = [report_metric]

    # Here we will save the final reports
    os.makedirs(args.save_path, exist_ok=True)
    pickle.dump(report_result, open(os.path.join(args.save_path, '{}_report.pkl'.format(args.task_name)),'wb'))
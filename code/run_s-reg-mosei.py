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

from sklearn.metrics import accuracy_score, f1_score

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
    
    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.float)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((s_inputs, s_attention_mask), batch_label, batch_name)
    

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_helper(pred_label, true_label):
    mae = np.mean(np.abs(pred_label - true_label))
    corr = np.corrcoef(pred_label, true_label)[0][1]
    
    nonzero_index = np.where(true_label!=0)
    pred_7class = np.clip(pred_label, a_min=-3., a_max=3.)
    true_7class = np.clip(true_label, a_min=-3., a_max=3.)
    pred_5class = np.clip(pred_label, a_min=-2., a_max=2.)
    true_5class = np.clip(true_label, a_min=-2., a_max=2.)

    acc_7class = multiclass_acc(pred_7class[nonzero_index],true_7class[nonzero_index])
    acc_5class = multiclass_acc(pred_5class[nonzero_index],true_5class[nonzero_index])

    pred_2class = pred_label[nonzero_index] > 0
    true_2class = true_label[nonzero_index] > 0

    acc_2class = accuracy_score(true_2class,pred_2class)
    f1_2class = f1_score(true_2class, pred_2class, average='weighted')

    return mae, corr, acc_7class, acc_5class, acc_2class, f1_2class


if __name__ == '__main__':

    data_root = '/dataset/mosei_mel160/V1_0/mel160/mel_160/'

    epoch_num = 20

    # Take Care, Here the model used MSE as the loss function
    # model = RobertaForSequenceClassification.from_pretrained(
    #     '/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-tiny/', num_labels=1)
    # model = RobertaForSequenceClassification.from_pretrained(
    #     '/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-base/', num_labels=1)
    model = RobertaForSequenceClassification.from_pretrained(
        '/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-large/', num_labels=1)
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    train_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/train.pkl','rb'))
    train_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in train_data]
        
    valid_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/dev.pkl','rb'))
    valid_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in valid_data]

    test_data = pickle.load(open('/dataset/mosei_mel160/V1_0/mel160/test.pkl','rb'))
    test_data = [(os.path.join(data_root,x[0]),x[1],x[2]) for x in test_data]

    batch_size = 32
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
    test_dataset = EmotionDataset(test_data, tokenizer)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x, tokenizer),
        shuffle = False, num_workers = num_workers
    )

    count, best_mae, best_epoch = 0, np.inf, 0
    report_metric = {}

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
            label_outputs = outputs.logits.view(-1).cpu().detach().numpy().astype(float)
            pred_y.extend(list(label_outputs))

        pred_y = np.array(pred_y)
        true_y = np.array(true_y)

        mae, corr, _, _, _, _ = eval_helper(pred_y, true_y)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("MAE: {:.3f} - Corr: {:.3f} - Train Loss: {:.3f} - Valid Loss: {:.3f}".format(mae, corr, epoch_train_loss, 0.0))

        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            print('Better MAE found on dev, calculate performance on Test')
            # Here we update the test result
            pred_y, true_y = [], []

            for semantic_inputs, label_inputs, _ in valid_loader:
                input_ids = semantic_inputs[0].cuda()
                attention_mask = semantic_inputs[1].cuda()
                true_y.extend(label_inputs.numpy())
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=None)
                label_outputs = outputs.logits.view(-1).cpu().detach().numpy().astype(float)
                pred_y.extend(list(label_outputs))

            pred_y = np.array(pred_y)
            true_y = np.array(true_y)

            mae, corr, acc_7, acc_5, acc_2, f1_2 = eval_helper(pred_y, true_y)
            report_metric = {'MAE':mae,'Corr':corr,'Acc_7':acc_7,'Acc_5':acc_5,'Acc_2':acc_2,'F1_2':f1_2}
            print("Test performance - MAE: {:.3f} - Corr: {:.3f}".format(mae, corr))

    print("End. Best epoch {:03d}: {}".format(best_epoch, str(report_metric)))
    # pickle.dump(report_metric, open('/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-tiny/reg-mosei_report.pkl','wb'))
    # pickle.dump(report_metric, open('/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-base/reg-mosei_report.pkl','wb'))
    pickle.dump(report_metric, open('/home/work/Projects/MMPretrain/code/result/result_language/libri-roberta-large/reg-mosei_report.pkl','wb'))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn

from transformers import RobertaTokenizer
from transformers import AdamW
from m2p_mask import process_test_MAM_data

from m2p_transformer import KYRegRobertaM2Transfer
from torch.nn.utils.rnn import pad_sequence
from functools import reduce

from sklearn.metrics import accuracy_score, f1_score
from eval_metrics_2 import eval_mosei_senti

import re
import math
import random
import time
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class EmotionDataset(object):
    def __init__(self, data_list, tokenizer, audio_length=None):
        self.data_list = data_list
        self.audio_length = audio_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        audio_name = re.sub('\.npy', '.wav', os.path.basename(audio_path))
        audio_input = torch.FloatTensor(np.load(audio_path))
        if self.audio_length is not None: audio_input = audio_input[:self.audio_length, :]
        text_words = [x.lower() for x in re.split(' +', re.sub('[\.,\?\!]', ' ', asr_text))]
        text_input = self.tokenizer(' '.join(text_words))
        return {'audio_input': audio_input, 'text_input': text_input, 'label': label, 'audio_name': audio_name}


def collate(sample_list, tokenizer, config):
    batch_audio = [x['audio_input'] for x in sample_list]
    pad_batch_audio = pad_sequence(batch_audio, batch_first=True)

    pad_batch_text = {
        'input_ids': [x['text_input']['input_ids'] for x in sample_list],
        'attention_mask': [x['text_input']['attention_mask'] for x in sample_list],
    }
    pad_batch_text = tokenizer.pad(pad_batch_text, return_tensors='pt')
    s_inputs = pad_batch_text['input_ids']
    s_attention_mask = pad_batch_text['attention_mask']

    a_attention_mask, a_inputs = process_test_MAM_data((pad_batch_audio,), config)

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.float)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((a_inputs, a_attention_mask),
            (s_inputs, s_attention_mask),
            batch_label, batch_name)


def mv_loss_fn(text_feature, acou_feature):
    '''
    *_feature: [batch, 762]
    '''

    text_norm = torch.norm(text_feature, p=2, dim=1)  # [batch]
    acou_norm = torch.norm(acou_feature, p=2, dim=1)

    mul_feat = torch.matmul(text_feature.unsqueeze(dim=1), acou_feature.unsqueeze(dim=-1)).squeeze()

    normed_mul_feat = mul_feat / (text_norm * acou_norm)

    mv_loss = torch.mean(torch.abs(normed_mul_feat))

    return mv_loss


def train(args, config, train_data, valid_data, test_data):
    ############################ PARAMETER SETTING ##########################
    num_workers = config['dataloader']['n_jobs']
    batch_size = config['dataloader']['batch_size']
    learning_rate = config['optimizer']['learning_rate']
    warmup_proportion = config['optimizer']['warmup_proportion']
    save_ckpt_dir = os.path.join(args.save_path, 'checkpoints')

    audio_length = 3000
    epochs = args.epochs

    tokenizer = RobertaTokenizer.from_pretrained(config['semantic']['tokenizer_path'])

    ############################## PREPARE DATASET ##########################
    train_dataset = EmotionDataset(train_data, tokenizer, audio_length)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, config['acoustic']),
        shuffle=True, num_workers=num_workers
    )
    valid_dataset = EmotionDataset(valid_data, tokenizer, audio_length)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, config['acoustic']),
        shuffle=False, num_workers=num_workers
    )
    test_dataset = EmotionDataset(test_data, tokenizer, audio_length)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, config['acoustic']),
        shuffle=False, num_workers=num_workers
    )
    ########################### CREATE MODEL #################################
    model = KYRegRobertaM2Transfer(args.ckpt_path, freeze=args.freeze)
    model.cuda()

    print("Loaded pretrained model from :", args.ckpt_path)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ########################### TRAINING #####################################
    count, best_mae, best_epoch = 0, np.inf, 0
    report_metric = {}

    for epoch in range(epochs):
        epoch_train_loss = []
        model.train()
        start_time = time.time()

        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for acoustic_inputs, semantic_inputs, label_inputs, _ in progress:
            a_inputs = acoustic_inputs[0].cuda()
            a_attention_mask = acoustic_inputs[1].cuda()
            s_inputs = semantic_inputs[0].cuda()
            s_attention_mask = semantic_inputs[1].cuda()

            label_inputs = label_inputs.cuda()

            model.zero_grad()
            logits, loss0, text_att, text_max, acou_att, acou_max = model(
                s_inputs=s_inputs,
                s_attention_mask=s_attention_mask,
                a_inputs=a_inputs,
                a_attention_mask=a_attention_mask,
                labels=label_inputs,
            )

            text_state = text_att + text_max
            acou_state = acou_att + acou_max

            mv_loss = mv_loss_fn(text_state, acou_state)

            loss = loss0

            epoch_train_loss.append(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            count += 1

            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, acc_train_loss))

        model.eval()
        pred_y, true_y, epoch_valid_loss = [], [], []

        with torch.no_grad():
            for acoustic_inputs, semantic_inputs, label_inputs, _ in valid_loader:
                a_inputs = acoustic_inputs[0].cuda()
                a_attention_mask = acoustic_inputs[1].cuda()
                s_inputs = semantic_inputs[0].cuda()
                s_attention_mask = semantic_inputs[1].cuda()

                true_y.append(label_inputs.cpu().detach())
                # label_inputs = label_inputs.cuda()

                logits, _, _, _, _, _ = model(
                    s_inputs=s_inputs,
                    s_attention_mask=s_attention_mask,
                    a_inputs=a_inputs,
                    a_attention_mask=a_attention_mask,
                    labels=None
                )

                # epoch_valid_loss.append(loss)

                pred_y.append(logits.cpu().detach())

        pred_y = torch.cat(pred_y, dim=0)
        true_y = torch.cat(true_y, dim=0)

        mae, corr, _, _, _, _ = eval_mosei_senti(results=pred_y, truths=true_y, exclude_zero=True, testing=False)

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
        # epoch_valid_loss = torch.mean(torch.tensor(epoch_valid_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("MAE: {:.3f} - Corr: {:.3f} - Train Loss: {:.3f} - Valid Loss: {:.3f}".format(mae, corr, epoch_train_loss,
                                                                                            0.0))

        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            print('Better MAE found on dev, calculate performance on Test')
            # Here we update the test result
            pred_y, true_y = [], []

            with torch.no_grad():
                for acoustic_inputs, semantic_inputs, label_inputs, _ in test_loader:
                    a_inputs = acoustic_inputs[0].cuda()
                    a_attention_mask = acoustic_inputs[1].cuda()
                    s_inputs = semantic_inputs[0].cuda()
                    s_attention_mask = semantic_inputs[1].cuda()

                    true_y.append(label_inputs.cpu().detach())

                    logits, _, _, _, _, _ = model(
                        s_inputs=s_inputs,
                        s_attention_mask=s_attention_mask,
                        a_inputs=a_inputs,
                        a_attention_mask=a_attention_mask,
                        labels=None
                    )

                    pred_y.append(logits.cpu().detach())

            pred_y = torch.cat(pred_y, dim=0)
            true_y = torch.cat(true_y, dim=0)

            mae, corr, acc_7, acc_5, f1_2, acc_2 = eval_mosei_senti(results=pred_y, truths=true_y, exclude_zero=True,
                                                                    testing=True)

            report_metric = {'MAE': mae, 'Corr': corr, 'Acc_7': acc_7, 'Acc_5': acc_5, 'Acc_2': acc_2, 'F1_2': f1_2}

            print("Test performance - MAE: {:.3f} - Corr: {:.3f}".format(mae, corr))

    print("End. Best epoch {:03d}: {}".format(best_epoch, str(report_metric)))
    return report_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("--config", type=str, default=None, help="config path")
    parser_train.add_argument("--epochs", type=int, default=5, help="training epoches")
    parser_train.add_argument("--save_path", type=str, default=None, help="ckpt save path")
    parser_train.add_argument("--ckpt_path", type=str, default=None, help="ckpt load path")
    parser_train.add_argument("--freeze", type=bool, default=False, help="freeze the pretrain model")
    parser_train.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    set_seed(args.seed)
    print("限制种子：", args.seed)

    data_root = './dataset/mosei_mel160/V1_0/mel160/mel_160/'

    if args.mode == 'train':
        save_path = args.save_path
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        train_data = pickle.load(open('./dataset/mosei_mel160/V1_0/mel160/train.pkl', 'rb'))
        train_data = [(os.path.join(data_root, x[0]), x[1], x[2]) for x in train_data]

        valid_data = pickle.load(open('./dataset/mosei_mel160/V1_0/mel160/dev.pkl', 'rb'))
        valid_data = [(os.path.join(data_root, x[0]), x[1], x[2]) for x in valid_data]

        test_data = pickle.load(open('./dataset/mosei_mel160/V1_0/mel160/test.pkl', 'rb'))
        test_data = [(os.path.join(data_root, x[0]), x[1], x[2]) for x in test_data]

        report_metric = train(args, config, train_data, valid_data, test_data)

        os.makedirs(args.save_path, exist_ok=True)

        pickle.dump(report_metric, open(os.path.join(args.save_path, 'reg-mosei_report.pkl'), 'wb'))

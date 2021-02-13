import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

import torch
import torch.nn as nn

from transformers import RobertaTokenizer
from transformers import AdamW
from m2p_mask import process_test_MAM_data

from m2p_transformer import RobertaM2Transfer

from torch.nn.utils.rnn import pad_sequence
from functools import reduce

import re
import random
import math
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


class SpeakerDataset(object):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, text_path = self.data_list[index]

        speaker = str(audio_path.split("/")[-1].split("-")[0])
        label = speaker2label[speaker]

        audio_path = os.path.join(data_root, audio_path)

        audio_name = re.sub('\.npy', '.wav', os.path.basename(audio_path))
        audio_input = torch.FloatTensor(np.load(audio_path))

        asr_text = ' '.join(
            [x.strip('\n').split(',')[2].lower() for x in open(os.path.join(data_root, text_path), 'r').readlines()])

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

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((a_inputs, a_attention_mask),
            (s_inputs, s_attention_mask),
            batch_label, batch_name)


def wa_helper(pred_label, true_label):
    result = np.mean(pred_label == true_label)
    return result


def ua_helper(pred_label, true_label, class_num=4):
    pred_onehot = np.eye(class_num)[pred_label]
    true_onehot = np.eye(class_num)[true_label]
    result = np.mean(np.sum((pred_onehot == true_onehot) * true_onehot, axis=0) / np.sum(true_onehot, axis=0))
    return result


def train(args, config, train_data):
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
    train_dataset = SpeakerDataset(train_data, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, config['acoustic']),
        shuffle=True, num_workers=num_workers
    )

    ########################### CREATE MODEL #################################
    model = RobertaM2Transfer(args.ckpt_path, args.speaker_nums, freeze=args.freeze, use_pretrained=False)
    model.cuda()
    model = torch.nn.DataParallel(model)

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
    count = 0

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
            logits, loss = model(
                s_inputs=s_inputs,
                s_attention_mask=s_attention_mask,
                a_inputs=a_inputs,
                a_attention_mask=a_attention_mask,
                labels=label_inputs,
            )

            loss = loss.mean()

            epoch_train_loss.append(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            count += 1

            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f}".format(epoch, acc_train_loss))


        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
        # epoch_valid_loss = torch.mean(torch.tensor(epoch_valid_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("Train Loss: {:.3f}".format(epoch_train_loss))

        torch.save(model.state_dict(), os.path.join(args.save_path, "model_{}.bin".format(epoch)))


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
    parser_train.add_argument("--speaker_nums", type=int, default=-1, help="numbers of speaker in train set")

    args = parser.parse_args()

    set_seed(args.seed)
    print("限制种子：", args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    train_data_source = [
        'train-clean-100_alignment.csv',
        'train-clean-360_alignment.csv',
        'train-other-500_alignment.csv'
    ]

    data_root = "./dataset/libri_mel160/V1_0/libri_mel160"

    speaker2label = pickle.load(open("./librispeech_train_speaker2id.pkl", 'rb'))

    if args.mode == 'train':

        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        train_data = [pd.read_csv(os.path.join(data_root, s)) for s in train_data_source]
        train_data = pd.concat(train_data, axis=0)

        train_data = train_data[train_data.length <= 1500]
        train_data = train_data.dropna().reset_index(drop=True)

        train_data = train_data[["file_path", "align_path"]].values.tolist()

        train(args, config, train_data)

        print("End.")


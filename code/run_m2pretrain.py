import os
import glob
import yaml
import torch
import random
import argparse
import numpy as np
from shutil import copyfile
from argparse import Namespace

from m2p_runner import Runner
from m2p_dataloader import MultiModalDataset
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer

def get_upstream_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for Upstream Models of the S3PLR project.')

    # required, set either (--run and --config) or (--resume)
    parser.add_argument('--run', default=None, choices=['transformer', 'apc'], help='Select pre-training task. \
                        For the transformer models, which type of pre-training (mockingjay, tera, aalbert, etc) \
                        is determined by config file.')
    parser.add_argument('--config', default=None, type=str, help='Path to experiment config.')
    parser.add_argument('--resume', default=None, help='Specify the upstream checkpoint path to resume training')

    # ckpt and logging
    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--ckpdir', default='', type=str, help='Path to store checkpoint result, if empty then default is used.')
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.')
    
    # Options
    parser.add_argument('--test', default='', type=str, help='Input path to the saved model ckpt for testing.')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable Multi-GPU training.')
    parser.add_argument('--test_reconstruct', action='store_true', help='Test reconstruction capability')
    parser.add_argument('--online_config', default=None, help='Explicitly specify the config of on-the-fly feature extraction')
    parser.add_argument('--kaldi_data', action='store_true', help='Whether to use the Kaldi dataset')

    # parse
    args = parser.parse_args()
    if args.resume is None:
        assert args.run is not None and args.config is not None, '`--run` and `--config` must be given if `--resume` is not provided'
        setattr(args, 'gpu', not args.cpu)
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        if args.online_config is not None:
            online_config = yaml.load(open(args.online_config, 'r'), Loader=yaml.FullLoader)
            config['online'] = online_config
    else:
        if os.path.isdir(args.resume):
            ckpts = glob.glob(f'{args.resume}/*.ckpt')
            assert len(ckpts) > 0
            ckpts = sorted(ckpts, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            resume_ckpt = ckpts[-1]
        else:
            resume_ckpt = args.resume

        def update_args(old, new):
            old_dict = vars(old)
            new_dict = vars(new)
            old_dict.update(new_dict)
            return Namespace(**old_dict)

        ckpt = torch.load(resume_ckpt, map_location='cpu')
        args = update_args(args, ckpt['Settings']['Paras'])
        config = ckpt['Settings']['Config']
        setattr(args, 'resume', resume_ckpt)        
    
    return args, config

def train(args, config):
    
    if args.ckpdir == '':
        if args.name is None: args.name = 'run_' + str(random.randint(0, 999))
        ckpdir = os.path.join('result/result_transformer/', args.name)
    else:
        ckpdir = args.ckpdir
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    copyfile(args.config, os.path.join(ckpdir, args.config.split('/')[-1]))

    tokenizer = RobertaTokenizer.from_pretrained(config['dataloader']['tokenizer_path'])

    dataset = MultiModalDataset(file_path=config['dataloader']['data_path'], sets=config['dataloader']['train_set'],
                                bucket_size=config['dataloader']['batch_size'],max_timestep=config['dataloader']['max_timestep'],
                                drop=True,acoustic_config=config['acoustic'],semantic_config=config['semantic'],tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False, 
                            num_workers=config['dataloader']['n_jobs'], pin_memory=True)
    runner = Runner(args, config, dataloader, ckpdir)
    runner.set_model()
    runner.train()


if __name__ == "__main__":
    args, config = get_upstream_args()
    train(args, config)
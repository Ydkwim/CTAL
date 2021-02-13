import os
import math

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import RobertaConfig

from m2p_model import RobertaM2Model
from m2p_optimization import BertAdam, Lamb, WarmupLinearSchedule

class Runner():
    ''' Handler for complete pre-training progress of upstream models '''
    def __init__(self, args, config, dataloader, ckpdir):
        
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(ckpdir)

        self.args = args
        self.config = config
        self.dataloader = dataloader
        self.ckpdir = ckpdir

        # optimizer
        self.learning_rate = float(config['optimizer']['learning_rate'])
        self.warmup_proportion = config['optimizer']['warmup_proportion']
        self.gradient_accumulation_steps = config['optimizer']['gradient_accumulation_steps']
        self.gradient_clipping = config['optimizer']['gradient_clipping']

        # Training details
        self.apex = config['runner']['apex']
        self.total_steps = config['runner']['total_steps']
        self.log_step = config['runner']['log_step']
        self.save_step = config['runner']['save_step']
        self.duo_feature = config['runner']['duo_feature']
        self.max_keep = config['runner']['max_keep']

        # Model configs
        self.semantic_config = RobertaConfig(**config['semantic'])
        self.acoustic_config = RobertaConfig(**config['acoustic'])

    def set_model(self):
        print('[Runner] - Initializing Transformer model...')
        self.model = RobertaM2Model(self.semantic_config, self.acoustic_config)
        self.model.cuda()
        self.model.train()

        if self.args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Runner] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        if 'type' not in self.config['optimizer']:
            self.config['optimizer']['type'] = 'adam'
        print('[Runner] - Optimizer: ' + ('apex Fused Adam' if self.apex else str(self.config['optimizer']['type'])))
        if self.apex:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=self.learning_rate,
                                    bias_correction=False,
                                    max_grad_norm=1.0)
            if self.config['optimizer']['loss_scale'] == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.config['optimizer']['loss_scale'])
            self.warmup_linear = WarmupLinearSchedule(warmup=self.warmup_proportion,
                                                      t_total=self.total_steps)
        elif self.config['optimizer']['type'] == 'adam':
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear')
        elif self.config['optimizer']['type'] == 'lamb' or self.config['optimizer']['type'] == 'adamW':
            self.optimizer = Lamb(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear',
                                      adam=True if self.config['optimizer']['type'] == 'adamW' else False,
                                      correct_bias=True if self.config['optimizer']['type'] == 'adamW' else False)
        else:
            raise NotImplementedError()
        
        if self.args.resume is not None:
            self.load_model(self.args.resume)

    def process_acoustic_data(self, acoustic_inputs):
        """Process training data for the masked acoustic model"""
        with torch.no_grad():
            
            assert(len(acoustic_inputs) == 4), 'dataloader should return (a_inputs, a_mask_labels, a_attn_mask, a_labels)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            a_inputs = acoustic_inputs[0].squeeze(0)
            a_mask_labels = acoustic_inputs[1].squeeze(0)
            a_attention_mask = acoustic_inputs[2].squeeze(0)
            a_labels = acoustic_inputs[3].squeeze(0)

            a_inputs = a_inputs.float().to(device=self.device)
            a_mask_labels = a_mask_labels.bool().to(device=self.device)
            a_attention_mask = a_attention_mask.float().to(device=self.device)
            a_labels = a_labels.float().to(device=self.device)

        return a_inputs, a_mask_labels, a_attention_mask, a_labels

    def process_semantic_data(self, semantic_inputs):
        with torch.no_grad():
            
            assert(len(semantic_inputs) == 3), 'dataloader should return (s_inputs, s_attention_mask, s_labels)'
            s_inputs = semantic_inputs[0].squeeze(0)
            s_attention_mask = semantic_inputs[1].squeeze(0)
            s_labels = semantic_inputs[2].squeeze(0)

            s_inputs = s_inputs.long().to(device=self.device)
            s_attention_mask = s_attention_mask.float().to(device=self.device)
            s_labels = s_labels.long().to(device=self.device)
        return s_inputs, s_attention_mask, s_labels

    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)
        self.model.semantic_model.load_state_dict(ckpt['semantic_model'])
        self.model.acoustic_model.load_state_dict(ckpt['acoustic_model'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']

    def save_model(self, name='states', to_path=None):
        all_states = {
            'semantic_model': self.model.semantic_model.state_dict() if not self.args.multi_gpu else self.model.module.semantic_model.state_dict(),
            'acoustic_model': self.model.acoustic_model.state_dict() if not self.args.multi_gpu else self.model.module.acoustic_model.state_dict(),
        }
        all_states['Optimizer'] = self.optimizer.state_dict()
        all_states['Global_step'] = self.global_step
        all_states['Settings'] = { 'Config': self.config, 'Paras': self.args }

        if to_path is None:
            new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
        else:
            new_model_path = to_path

        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def train(self,):
        pbar = tqdm(total=self.total_steps)
        pbar.n = self.global_step - 1
        while self.global_step <= self.total_steps:
            progress = tqdm(self.dataloader, desc="Iteration")

            loss_val, acoustic_loss_val, semantic_loss_val = 0, 0, 0
            for batch in progress:
                batch_is_valid, acoustic_batch, semantic_batch = batch
                try:
                    if self.global_step > self.total_steps: break
                    if not batch_is_valid: continue

                    a_inputs, a_mask_labels, a_attention_mask, a_labels = self.process_acoustic_data(acoustic_batch)
                    s_inputs, s_attention_mask, s_labels = self.process_semantic_data(semantic_batch)

                    semantic_loss, acoustic_loss, _, _ = self.model(s_inputs=s_inputs,s_attention_mask=s_attention_mask,s_labels=s_labels,
                                                                    a_inputs=a_inputs,a_attention_mask=a_attention_mask,a_labels=a_labels,a_mask_labels=a_mask_labels)
                    
                    loss = semantic_loss + acoustic_loss

                    # Accumulate Loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    if self.apex and self.args.multi_gpu:
                        raise NotImplementedError
                    elif self.apex:
                        self.optimizer.backward(loss)
                    elif self.args.multi_gpu:
                        acoustic_loss = acoustic_loss.mean()
                        semantic_loss = semantic_loss.mean()
                        loss = loss.mean()
                        loss.backward()
                    else:
                        loss.backward()
                    
                    loss_val += loss.item()
                    acoustic_loss_val += acoustic_loss.item()
                    semantic_loss_val += semantic_loss.item()

                    if (self.total_steps+1) % self.gradient_accumulation_steps == 0:
                        if self.apex:
                            # modify learning rate with special warm up BERT uses
                            # if conifg.apex is False, BertAdam is used and handles this automatically
                            lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr_this_step

                        # Step
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                        if math.isnan(grad_norm):
                            print('[Runner] - Error : grad norm is NaN @ step ' + str(self.global_step))
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.global_step % self.log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
                            self.log.add_scalar('loss', (loss_val), self.global_step)
                            self.log.add_scalar('spec_loss', (acoustic_loss_val), self.global_step)
                            self.log.add_scalar('text_loss', (semantic_loss_val), self.global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        progress.set_description("Loss {:.4f} - Spec Loss {:.4f} - Text Loss {:.4f}".format(loss_val, acoustic_loss_val, semantic_loss_val))

                        if self.global_step % self.save_step == 0:
                            self.save_model('states')

                        loss_val, acoustic_loss_val, semantic_loss_val = 0, 0, 0
                        pbar.update(1)
                        self.global_step += 1

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise
        pbar.close()
        self.log.close()
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel
from m2p_model import AcousticModel


class RobertaM2Upstream(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        self.acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        self.semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(self.acoustic_config)
        semantic_model = RobertaModel(self.semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

    def load_model(self, transformer_model, state_dict, prefix_name=''):
        try:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(transformer_model, prefix_name)
            if len(missing_keys) > 0:
                print('Weights of {} not initialized from pretrained model: {}'.format(
                    transformer_model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print('Weights from pretrained model not used in {}: {}'.format(
                    transformer_model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    transformer_model.__class__.__name__, '\n\t'.join(error_msgs)))
            print('[CTAL] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[CTAL] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

    def _forward(self,
                 s_inputs=None,
                 s_attention_mask=None,
                 s_token_type_ids=None,
                 s_position_ids=None,
                 s_head_mask=None,
                 a_inputs=None,
                 a_attention_mask=None,
                 a_token_type_ids=None,
                 a_position_ids=None,
                 a_head_mask=None,
                 output_attentions=False,
                 output_hidden_states=False,
                 return_dict=None):
        
        semantic_outputs = self.semantic_model(
            s_inputs,
            attention_mask=s_attention_mask,
            token_type_ids=s_token_type_ids,
            position_ids=s_position_ids,
            head_mask=s_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        semantic_encode = semantic_outputs[0]

        acoustic_outputs = self.acoustic_model(
            a_inputs,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=a_position_ids,
            head_mask=a_head_mask,
            encoder_hidden_states=semantic_encode,
            encoder_attention_mask=s_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        acoustic_encode = acoustic_outputs[0]

        return semantic_encode, acoustic_encode


class RobertaM2Downstream(RobertaM2Upstream):
    def __init__(self, ckpt_path, label_num, drop_rate=0.2, freeze=False, orthogonal_fusion=True, head='both'):
        assert head in ['both','acoustic','semantic']
        super().__init__(ckpt_path)
        self.head = head
        self.freeze = freeze
        self.label_num = label_num
        self.orthogonal_fusion = (orthogonal_fusion and self.head=='both')
        
        if self.head in ['both','acoustic']:
            self.acoustic_linear = nn.Linear(self.acoustic_config.hidden_size,self.acoustic_config.hidden_size)
            self.acoustic_attention = nn.Linear(self.acoustic_config.hidden_size,1)
        if self.head in ['both','semantic']:
            self.semantic_linear = nn.Linear(self.semantic_config.hidden_size,self.semantic_config.hidden_size)

        self.fuse_linear = nn.Linear(self.acoustic_config.hidden_size+self.semantic_config.hidden_size, 
                                     self.acoustic_config.hidden_size+self.semantic_config.hidden_size)

        self.classifier = nn.Linear(self.acoustic_config.hidden_size+self.semantic_config.hidden_size, 
                                    self.label_num, bias=False)

        self.dropout = nn.Dropout(drop_rate)

        if self.label_num == 1:
            self.loss_fct = nn.L1Loss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, 
                s_inputs=None,
                s_attention_mask=None,
                s_token_type_ids=None,
                s_position_ids=None,
                s_head_mask=None,
                a_inputs=None,
                a_attention_mask=None,
                a_token_type_ids=None,
                a_position_ids=None,
                a_head_mask=None,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None):
        
        if self.freeze:
            with torch.no_grad():
                semantic_encode, acoustic_encode = self._forward(
                    s_inputs,
                    s_attention_mask,
                    s_token_type_ids,
                    s_position_ids,
                    s_head_mask,
                    a_inputs,
                    a_attention_mask,
                    a_token_type_ids,
                    a_position_ids,
                    a_head_mask,
                    output_attentions,
                    output_hidden_states,
                    return_dict)
        else:
            semantic_encode, acoustic_encode = self._forward(
                s_inputs,
                s_attention_mask,
                s_token_type_ids,
                s_position_ids,
                s_head_mask,
                a_inputs,
                a_attention_mask,
                a_token_type_ids,
                a_position_ids,
                a_head_mask,
                output_attentions,
                output_hidden_states,
                return_dict)

        if self.head in ['both','semantic']:
            semantic_encode = self.semantic_linear(semantic_encode)
            semantic_encode = torch.tanh(semantic_encode)

            semantic_pool = semantic_encode[:,0,:]

            semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
            semantic_max = torch.max(semantic_encode[:,1:,:],dim=1)[0]

        if self.head in ['both','acoustic']:
            acoustic_encode = self.acoustic_linear(acoustic_encode)
            acoustic_encode = torch.tanh(acoustic_encode)

            acoustic_pool = self.acoustic_attention(acoustic_encode)
            acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
            acoustic_pool = F.softmax(acoustic_pool, dim=1)
            acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

            acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
            acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        if self.head == 'both':
            if self.orthogonal_fusion:
                att_feats = semantic_pool + acoustic_pool
                max_feats = semantic_max + acoustic_max
                fuse_encode = torch.cat([att_feats, max_feats], dim=-1)
            else:
                fuse_encode = torch.cat([semantic_pool, acoustic_pool],dim=-1)

        elif self.head == 'semantic':
            fuse_encode = torch.cat([semantic_pool,semantic_max], dim=-1)
        else:
            fuse_encode = torch.cat([acoustic_pool,acoustic_max], dim=-1)

        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)

        logits = self.classifier(fuse_encode_act)
        
        if labels is not None:
            if self.label_num == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fct(logits, labels.long())

            if self.orthogonal_fusion:

                orth_loss_attn = self.orthogonal_loss(acoustic_pool, semantic_pool)
                orth_loss_max = self.orthogonal_loss(acoustic_max, semantic_max)
                orth_loss = orth_loss_attn + orth_loss_max

                loss += orth_loss

        else:
            loss = None

        return fuse_encode_act, logits, loss

    def orthogonal_loss(self, acoustic_state, semantic_state):

        acoustic_norm = torch.norm(acoustic_state, p=2, dim=1)
        semantic_norm = torch.norm(acoustic_state, p=2, dim=1)

        orth_loss = torch.diag(torch.matmul(acoustic_state, semantic_state.permute(1, 0)))
        orth_loss = torch.mean(torch.abs(orth_loss / (acoustic_norm * semantic_norm)))

        return orth_loss




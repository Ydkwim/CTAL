import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import RobertaConfig, RobertaModel
from m2p_model import AcousticModel

# Build the model suitable for the further finetuning
class RobertaM2Transfer(nn.Module):
    def __init__(self, ckpt_path, class_num, drop_rate=0.1, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(acoustic_config)
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        self.acoustic_linear = nn.Linear(acoustic_config.hidden_size,acoustic_config.hidden_size)
        self.semantic_linear = nn.Linear(semantic_config.hidden_size,semantic_config.hidden_size)
        
        # build the classifier based one attention pooling and max pooling
        self.acoustic_attention = nn.Linear(acoustic_config.hidden_size,1)
        
        self.fuse_linear = nn.Linear(acoustic_config.hidden_size+semantic_config.hidden_size, acoustic_config.hidden_size+semantic_config.hidden_size)

        self.classifier = nn.Linear(acoustic_config.hidden_size+semantic_config.hidden_size, class_num, bias=False)
        
        self.dropout = nn.Dropout(drop_rate)

        self.loss_fct = nn.CrossEntropyLoss()

        
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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

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
                 return_dict=None
                ):
        
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
                    return_dict
                )
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
                return_dict
            )
        # Since Here we will need to performe the interest process
        semantic_encode = self.semantic_linear(semantic_encode)
        semantic_encode = torch.tanh(semantic_encode)
        
        semantic_pool = semantic_encode[:,0,:]
        
        semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
        semantic_max = torch.max(semantic_encode, dim=1)[0]

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)
        
        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        # we first try with the concat
        fuse_encode = torch.cat([semantic_pool, acoustic_pool],dim=-1)
        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)

        logits = self.classifier(fuse_encode_act)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss




class RobertaM2TransferText(nn.Module):
    def __init__(self, ckpt_path, class_num, drop_rate=0.1, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        self.linear = nn.Linear(semantic_config.hidden_size, semantic_config.hidden_size)
        
        self.classifier = nn.Linear(semantic_config.hidden_size, class_num)
        
        self.dropout = nn.Dropout(drop_rate)

        self.loss_fct = nn.CrossEntropyLoss()

        
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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

    def _forward(self, 
                 s_inputs=None,
                 s_attention_mask=None,
                 s_token_type_ids=None,
                 s_position_ids=None,
                 s_head_mask=None,
                 output_attentions=False,
                 output_hidden_states=False,
                 return_dict=None
                ):
        
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

        return semantic_encode

    def forward(self, 
                s_inputs=None,
                s_attention_mask=None,
                s_token_type_ids=None,
                s_position_ids=None,
                s_head_mask=None,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None):
        
        if self.freeze:
            with torch.no_grad():
                semantic_encode = self._forward(
                    s_inputs,
                    s_attention_mask,
                    s_token_type_ids,
                    s_position_ids,
                    s_head_mask,
                    output_attentions,
                    output_hidden_states,
                    return_dict
                )
        else:
            semantic_encode = self._forward(
                s_inputs,
                s_attention_mask,
                s_token_type_ids,
                s_position_ids,
                s_head_mask,
                output_attentions,
                output_hidden_states,
                return_dict
            )
        # Since Here we will need to performe the interest process
        semantic_pool = semantic_encode[:,0,:]
        semantic_pool = self.dropout(semantic_pool)
        
        semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
        semantic_max = torch.max(semantic_encode, dim=1)[0]

        semantic_linear = self.linear(semantic_pool)
        semantic_linear = torch.tanh(semantic_linear)
        
        semantic_linear = self.dropout(semantic_linear)
        logits = self.classifier(semantic_linear)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss



class RobertaM2TransferAcoustic(nn.Module):
    def __init__(self, ckpt_path, class_num, drop_rate=0.1, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(acoustic_config)
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        # build the classifier based one attention pooling and max pooling
        self.acoustic_attention = nn.Linear(acoustic_config.hidden_size,1)
        
        self.linear = nn.Linear(acoustic_config.hidden_size, acoustic_config.hidden_size)

        self.classifier = nn.Linear(acoustic_config.hidden_size, class_num, bias=False)
        
        self.dropout = nn.Dropout(drop_rate)

        self.loss_fct = nn.CrossEntropyLoss()

        
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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

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
                 return_dict=None
                ):
        
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
                _, acoustic_encode = self._forward(
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
                    return_dict
                )
        else:
            _, acoustic_encode = self._forward(
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
                return_dict
            )
        # Since Here we will need to performe the interest process

        acoustic_encode = self.linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)

        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        logits = self.classifier(acoustic_pool)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss



# This is set for pretrained model from S3PR
from transformer.nn_transformer import TRANSFORMER

class PretrainM2Transfer(nn.Module):
    def __init__(self, class_num, drop_rate=0.1):
        super().__init__()
        options = {
            'load_pretrain' : 'True',
            'dropout'       : 'default',
            'spec_aug'      : 'False',
            'spec_aug_prev' : 'True',
            'permute_input' : 'False',
            'no_grad'       : 'False',
            'weighted_sum'  : 'False',
            'select_layer'  : -1,
            'ckpt_file'     : '/share/HangLi/S3PRL/pretrained/MelBaseM6-libri/states-500000.ckpt'
        }
        self.acoustic_model = TRANSFORMER(options=options, inp_dim=160)
        self.semantic_model = RobertaModel.from_pretrained(
            '/share/HangLi/Language/data/libri-roberta-small-3layer/',add_pooling_layer=False)
        self.acoustic_linear = nn.Linear(768,768)
        self.semantic_linear = nn.Linear(768,768)

        self.acoustic_attention = nn.Linear(768,1)

        self.fuse_linear = nn.Linear(1536, 1536)
        self.classifier = nn.Linear(1536, class_num, bias=False)

        self.dropout = nn.Dropout(drop_rate)
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
        
        semantic_outputs = self.semantic_model(
            input_ids=s_inputs,
            attention_mask=s_attention_mask,
            token_type_ids=s_token_type_ids,
            position_ids=s_position_ids,
            head_mask=s_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        semantic_encode = semantic_outputs[0]

        acoustic_encode = self.acoustic_model(
            x=a_inputs
        )

        semantic_encode = self.semantic_linear(semantic_encode)
        semantic_encode = torch.tanh(semantic_encode)
        
        semantic_pool = semantic_encode[:,0,:]
        
        semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
        semantic_max = torch.max(semantic_encode, dim=1)[0]

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)
        
        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        # we first try with the concat
        fuse_encode = torch.cat([semantic_pool, acoustic_pool],dim=-1)
        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)

        logits = self.classifier(fuse_encode_act)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss



class PretrainAcousticTransfer(nn.Module):
    def __init__(self, class_num, drop_rate=0.1):
        super().__init__()
        options = {
            'load_pretrain' : 'True',
            'dropout'       : 'default',
            'spec_aug'      : 'False',
            'spec_aug_prev' : 'True',
            'permute_input' : 'False',
            'no_grad'       : 'False',
            'weighted_sum'  : 'False',
            'select_layer'  : -1,
            'ckpt_file'     : '/share/HangLi/S3PRL/pretrained/MelBaseM6-libri/states-500000.ckpt'
        }
        self.acoustic_model = TRANSFORMER(options=options, inp_dim=160)
        self.acoustic_linear = nn.Linear(768,768)

        self.acoustic_attention = nn.Linear(768,1)

        self.classifier = nn.Linear(768, class_num, bias=False)

        self.dropout = nn.Dropout(drop_rate)
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, 
                a_inputs=None,
                a_attention_mask=None,
                a_token_type_ids=None,
                a_position_ids=None,
                a_head_mask=None,
                labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None):

        acoustic_encode = self.acoustic_model(
            x=a_inputs
        )

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)
        
        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        logits = self.classifier(acoustic_pool)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss


class KYRobertaM2Transfer(nn.Module):
    def __init__(self, ckpt_path, class_num, drop_rate=0.1, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(acoustic_config)
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        self.acoustic_linear = nn.Linear(acoustic_config.hidden_size,acoustic_config.hidden_size)
        self.semantic_linear = nn.Linear(semantic_config.hidden_size,semantic_config.hidden_size)
        
        # build the classifier based one attention pooling and max pooling
        self.acoustic_attention = nn.Linear(acoustic_config.hidden_size,1)

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fuse_linear = nn.Linear((acoustic_config.hidden_size+semantic_config.hidden_size),
                                     (acoustic_config.hidden_size+semantic_config.hidden_size))

        self.classifier = nn.Linear((acoustic_config.hidden_size+semantic_config.hidden_size), class_num, bias=False)
        
        self.dropout = nn.Dropout(drop_rate)

        self.loss_fct = nn.CrossEntropyLoss()

        
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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

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
                 return_dict=None
                ):
        
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
                    return_dict
                )
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
                return_dict
            )
        # Since Here we will need to performe the interest process
        semantic_encode = self.semantic_linear(semantic_encode)
        semantic_encode = torch.tanh(semantic_encode)
        
        text_att = semantic_encode[:,0,:] #[cls]
        
        semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
        text_max = self.max_pool(semantic_encode[:, 1:, :].transpose(1, 2)).squeeze(dim=-1)

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)
        
        acoustic_pool = self.acoustic_attention(acoustic_encode) #att
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = self.max_pool(acoustic_encode.transpose(1, 2)).squeeze(dim=-1)

        # we first try with the concat

        att_feats = text_att + acoustic_pool
        max_feats = text_max + acoustic_max

        fuse_encode = torch.cat([att_feats, max_feats],dim=-1)
        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)

        logits = self.classifier(fuse_encode_act)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None

        return logits, loss, text_att, acoustic_pool, text_max, acoustic_max




class RegRobertaM2Transfer(nn.Module):
    def __init__(self, ckpt_path, drop_rate=0.1, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(acoustic_config)
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        self.acoustic_linear = nn.Linear(acoustic_config.hidden_size,acoustic_config.hidden_size)
        self.semantic_linear = nn.Linear(semantic_config.hidden_size,semantic_config.hidden_size)
        
        # build the classifier based one attention pooling and max pooling
        self.acoustic_attention = nn.Linear(acoustic_config.hidden_size,1)
        
        self.fuse_linear = nn.Linear(acoustic_config.hidden_size+semantic_config.hidden_size, acoustic_config.hidden_size+semantic_config.hidden_size)

        self.regressor = nn.Linear(acoustic_config.hidden_size+semantic_config.hidden_size, 1, bias=False)

        self.scaler_a = nn.Parameter(torch.FloatTensor([6.0]), requires_grad=False)
        
        self.scaler_b = nn.Parameter(torch.FloatTensor([-3.0]), requires_grad=False)
        
        self.dropout = nn.Dropout(drop_rate)

        # self.regressor_act = nn.Tanh()

        self.regressor_act = nn.Sigmoid()

        self.loss_fct = nn.L1Loss()

        
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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except: 
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

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
                 return_dict=None
                ):
        
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
                    return_dict
                )
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
                return_dict
            )
        # Since Here we will need to performe the interest process
        semantic_encode = self.semantic_linear(semantic_encode)
        semantic_encode = torch.tanh(semantic_encode)
        
        semantic_pool = semantic_encode[:,0,:]
        
        semantic_encode = semantic_encode * s_attention_mask[:,:,None] - 10000.0 * (1.0 - s_attention_mask[:,:,None])
        semantic_max = torch.max(semantic_encode, dim=1)[0]

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)
        
        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0,2,1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:,:,None] - 10000.0 * (1.0 - a_attention_mask[:,:,None])
        acoustic_max = torch.max(acoustic_encode, dim=1)[0]

        # we first try with the concat
        fuse_encode = torch.cat([semantic_pool, acoustic_pool],dim=-1)
        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)


        logits = self.regressor(fuse_encode_act)[:,0]
        # logits = self.regressor_act(logits) * self.scaler_a + self.scaler_b

        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss = None

        return logits, loss




class KYRegRobertaM2Transfer(nn.Module):
    def __init__(self, ckpt_path, drop_rate=0.2, freeze=False):
        super().__init__()
        self.freeze = freeze
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(acoustic_config)
        semantic_model = RobertaModel(semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

        self.acoustic_linear = nn.Linear(acoustic_config.hidden_size, acoustic_config.hidden_size)
        self.semantic_linear = nn.Linear(semantic_config.hidden_size, semantic_config.hidden_size)

        # build the classifier based one attention pooling and max pooling
        self.acoustic_attention = nn.Linear(acoustic_config.hidden_size, 1)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fuse_linear = nn.Linear((acoustic_config.hidden_size+semantic_config.hidden_size),
                                     (acoustic_config.hidden_size+semantic_config.hidden_size))

        self.regressor = nn.Linear((acoustic_config.hidden_size+semantic_config.hidden_size), 1, bias=False)

        self.scaler_a = nn.Parameter(torch.FloatTensor([6.0]), requires_grad=False)

        self.scaler_b = nn.Parameter(torch.FloatTensor([-3.0]), requires_grad=False)

        self.dropout = nn.Dropout(drop_rate)

        # self.regressor_act = nn.Tanh()

        self.regressor_act = nn.Sigmoid()

        self.loss_fct = nn.L1Loss()

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
            print('[Transformer] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except:
            raise RuntimeError('[Transformer] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

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
                 return_dict=None
                 ):

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
                    return_dict
                )
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
                return_dict
            )
        # Since Here we will need to performe the interest process
        semantic_encode = self.semantic_linear(semantic_encode)
        semantic_encode = torch.tanh(semantic_encode)

        text_att = semantic_encode[:, 0, :]

        semantic_encode = semantic_encode * s_attention_mask[:, :, None] - 10000.0 * (
                    1.0 - s_attention_mask[:, :, None])
        text_max = self.max_pool(semantic_encode[:, 1:, :].transpose(1, 2)).squeeze(dim=-1)

        acoustic_encode = self.acoustic_linear(acoustic_encode)
        acoustic_encode = torch.tanh(acoustic_encode)

        acoustic_pool = self.acoustic_attention(acoustic_encode)
        acoustic_pool = acoustic_pool * a_attention_mask[:, :, None] - 10000.0 * (1.0 - a_attention_mask[:, :, None])
        acoustic_pool = F.softmax(acoustic_pool, dim=1)
        acoustic_pool = torch.matmul(acoustic_pool.permute(0, 2, 1), acoustic_encode).squeeze(1)

        acoustic_encode = acoustic_encode * a_attention_mask[:, :, None] - 10000.0 * (
                    1.0 - a_attention_mask[:, :, None])
        acoustic_max = self.max_pool(acoustic_encode.transpose(1, 2)).squeeze(dim=-1)

        # we first try with the concat

        att_feats = text_att + acoustic_pool
        max_feats = text_max + acoustic_max

        fuse_encode = torch.cat([att_feats, max_feats], dim=-1)
        fuse_encode = self.fuse_linear(fuse_encode)
        fuse_encode_act = torch.tanh(fuse_encode)

        logits = self.regressor(fuse_encode_act)[:, 0]
        # logits = self.regressor_act(logits) * self.scaler_a + self.scaler_b

        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss = None

        return logits, loss, text_att, text_max, acoustic_pool, acoustic_max







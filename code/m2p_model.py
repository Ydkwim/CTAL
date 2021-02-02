import os
import sys
import torch
import torch.nn as nn

from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, MaskedLMOutput

from transformers.activations import ACT2FN, gelu
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AcousticInputRepresentations(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.dense = nn.Linear(input_dim * config.downsample_rate, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]
        spec_transformed = self.dense(inputs_embeds)
        
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        input_representations = spec_transformed + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            input_representations = input_representations + position_embeddings

        input_representations = self.LayerNorm(input_representations)
        input_representations = self.dropout(input_representations)
        return input_representations


class RobertaAMHead(nn.Module):
    """Roberta Head for masked audio modeling."""
    def __init__(self, config, output_dim, input_dim=None):
        super().__init__()
        self.output_dim = output_dim
        if input_dim is None:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(input_dim, config.hidden_size)

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, self.output_dim*config.downsample_rate, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim*config.downsample_rate))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.act_fn(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class AcousticModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = AcousticInputRepresentations(config, config.audio_size)
        self.encoder = RobertaEncoder(config)
        self.pooler = None
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        input_shape = inputs_embeds.size()[:-1]

        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class RobertaForMaskedAM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    def __init__(self, config):
        super().__init__(config)
        if not config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedAM` make sure `config.is_decoder=True` for "
                "bi-directional decoder attention."
            )
        
        self.acoustic = AcousticModel(config)
        self.am_head = RobertaAMHead(config, config.audio_size)

        self.init_weights()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        mask_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.acoustic(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.am_head(sequence_output)

        masked_am_loss = None
        if labels is not None:
            loss_fct = nn.L1Loss()
            masked_am_loss = loss_fct(prediction_scores.masked_select(mask_labels), labels.masked_select(mask_labels))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_am_loss,) + output) if masked_am_loss is not None else output

        return MaskedLMOutput(
            loss=masked_am_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class RobertaM2Model(nn.Module):
    def __init__(self, semantic_config, acoustic_config):
        super().__init__()
        self.semantic_config = semantic_config
        self.acoustic_config = acoustic_config
        self.semantic_model = RobertaForMaskedLM(semantic_config)
        self.acoustic_model = RobertaForMaskedAM(acoustic_config)

    def forward(self, 
                s_inputs=None,
                s_attention_mask=None,
                s_token_type_ids=None,
                s_position_ids=None,
                s_head_mask=None,
                s_labels=None,
                a_inputs=None,
                a_attention_mask=None,
                a_token_type_ids=None,
                a_position_ids=None,
                a_head_mask=None,
                a_labels=None,
                a_mask_labels=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None):
        
        return_dict = return_dict if return_dict is not None else (self.acoustic_config.use_return_dict and self.semantic_config.use_return_dict)

        semantic_outputs = self.semantic_model(
            s_inputs,
            attention_mask=s_attention_mask,
            token_type_ids=s_token_type_ids,
            position_ids=s_position_ids,
            head_mask=s_head_mask,
            labels=s_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        semantic_loss = semantic_outputs.loss
        # Take the last layers' hidden
        semantic_encode = semantic_outputs.hidden_states[-1]

        acoustic_outputs = self.acoustic_model(
            a_inputs,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=a_position_ids,
            head_mask=a_head_mask,
            encoder_hidden_states=semantic_encode,
            encoder_attention_mask=s_attention_mask,
            labels=a_labels,
            mask_labels=a_mask_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        acoustic_loss = acoustic_outputs.loss
        acoustic_encode = acoustic_outputs.hidden_states[-1]

        return semantic_loss, acoustic_loss, semantic_encode, acoustic_encode
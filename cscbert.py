import math
import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertModel, BertOnlyMLMHead, BertPreTrainedModel


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.transform = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.candidates_weight = nn.Linear(config.hidden_size * 4, config.hidden_size)

        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fuse_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, layer_output, candidates_embeddings, candidates_mask):
        """
        :param layer_output: [bsz, seq_len, hidden_dim]
        :param candidates_embeddings: [bsz, seq_len, candidates_num, hidden_dim]
        :param candidates_mask: [bsz, seq_len, candidates_num]
        :return:
        """
        bsz, seq_len, hidden_dim = layer_output.size()
        layer_output = layer_output.view(bsz * seq_len, 1, hidden_dim)
        candidates_embeddings = candidates_embeddings.view(bsz * seq_len, -1, hidden_dim)
        q = self.query(layer_output).view(bsz * seq_len, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        k = self.key(candidates_embeddings).view(bsz * seq_len, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        v = self.value(candidates_embeddings).view(bsz * seq_len, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))  # (bsz * seqlen, num_attention_heads, 1, candidates_num)
        attn_scores = attn_scores / math.sqrt(self.attention_head_size)

        candidates_mask = candidates_mask.view(bsz * seq_len, -1)   # (bsz * seqlen, candidates_num)
        attn_scores.masked_fill_(candidates_mask.unsqueeze(1).unsqueeze(2), 1e-10)
        attn_weight = torch.softmax(attn_scores, dim=-1)    # (bsz * seqlen, num_attention_heads, 1, candidates_num)
        attn_weight = self.dropout(attn_weight)

        context = torch.matmul(attn_weight, v)  # (bsz * seqlen, num_attention_heads, 1, attention_head_size)
        context = context.view(bsz * seq_len, -1)
        context.masked_fill_(candidates_mask.all(dim=-1, keepdim=True), 0.0)

        # context = self.layer_norm(context)

        context = self.transform(context)
        context = self.dropout(self.act(context))
        context = self.candidates_weight(context)
        
        context = context.view(bsz, seq_len, -1)
        layer_output = layer_output.view(bsz, seq_len, -1)
        layer_output = context + layer_output

        layer_output = self.dropout(layer_output)
        layer_output = self.fuse_layernorm(layer_output)
        return layer_output


class CSCBert(BertPreTrainedModel):
    def __init__(self, config):
        super(CSCBert, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        embedding_size = self.config.to_dict()['hidden_size']
        bert_embedding = self.bert.embeddings
        word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        embeddings = nn.Parameter(word_embeddings_weight, True)
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size, embedding_size, _weight=embeddings)
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings

        self.adapter = Adapter(config)
  
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, segment_ids, candidates_ids, label, inject_position=[1]):
        # input_ids: bsz * seq len
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.bert.embeddings(input_ids=input_ids, token_type_ids=segment_ids)

       
        candidates_mask = candidates_ids.eq(0)
        candidates_ids = candidates_ids.view(batch_size * seq_length, -1)
        candidates_embeddings = self.bert.embeddings.word_embeddings(candidates_ids)
        candidates_embeddings = candidates_embeddings.view(batch_size, seq_length, -1, self.config.hidden_size)
     
        hidden_states = embedding_output

        for i, layer_module in enumerate(self.bert.encoder.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if i in inject_position:
                hidden_states = self.adapter(hidden_states, candidates_embeddings, candidates_mask)
                
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                )
            hidden_states = layer_outputs[0]

        sequence_output = hidden_states
        prediction_scores = self.linear(sequence_output)   # bsz * seq len * vocab size
        prob = F.log_softmax(prediction_scores, dim=-1) # bsz * seq len * vocab size
        loss = F.nll_loss(prob.view(-1, self.config.vocab_size), label.view(-1), ignore_index=0)
        return loss, prob

    
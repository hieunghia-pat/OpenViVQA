import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, RobertaModel

from data_utils.vocab import Vocab
from models.modules.encoders import EncoderLayer
from models.utils import generate_sequential_mask, sinusoid_encoding_table, generate_padding_mask
from models.modules.containers import Module

class BERTModel(Module):
    def __init__(self, vocab: Vocab, pretrained_language_model_name, language_model_hidden_size=768,
                    d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, max_len=54, dropout=.1):
        super(BERTModel, self).__init__()
        self.padding_idx = vocab.padding_idx
        self.d_model = d_model

        self.language_model = BertModel.from_pretrained(pretrained_language_model_name, return_dict=True)
        # frozen the language model
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        self.proj_to_caption_model = nn.Linear(language_model_hidden_size, d_model)

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, padding_idx=0), freeze=True)
        self.encoder_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.proj_to_vocab = nn.Linear(d_model, len(vocab))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        b_s, seq_len = input_ids.shape
        mask_queries = generate_padding_mask(input_ids, self.padding_idx).to(input_ids.device)  # (b_s, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len).to(input_ids.device)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = torch.logical_or(mask_self_attention, mask_queries.unsqueeze(1).unsqueeze(1))
                
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input_ids.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(bool).to(input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).long().to(input_ids.device)

        bert_output = self.language_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        language_feature = self.proj_to_caption_model(bert_output.last_hidden_state)
        language_feature = language_feature + self.pos_emb(seq)

        # fine tuning the pretrained BERT-based model
        language_feature = self.encoder_layer(queries=language_feature,
                                                keys=language_feature, 
                                                values=language_feature,
                                                padding_mask=mask_queries.unsqueeze(-1),
                                                attention_mask=mask_self_attention)

        logits = self.proj_to_vocab(language_feature)
        out = F.log_softmax(logits, dim=-1)
        return out, language_feature

class PhoBERTModel(Module):
    def __init__(self, vocab: Vocab, pretrained_language_model_name, language_model_hidden_size=768,
                    d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, max_len=54, dropout=.1):
        super(PhoBERTModel, self).__init__()
        self.vocab = vocab
        self.padding_idx = vocab.padding_idx
        self.d_model = d_model

        self.language_model = RobertaModel.from_pretrained(pretrained_language_model_name, return_dict=True)
        # frozen the language model
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        self.proj_to_caption_model = nn.Linear(language_model_hidden_size, d_model)

        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, padding_idx=0), freeze=True)
        self.encoder_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.proj_to_vocab = nn.Linear(d_model, len(vocab))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        '''
        Forward language model.
        '''
        b_s, seq_len = input_ids.shape
        mask_queries = generate_padding_mask(input_ids, self.padding_idx).to(input_ids.device)  # (b_s, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len).to(input_ids.device)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = torch.logical_or(mask_self_attention, mask_queries.unsqueeze(1).unsqueeze(1))

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input_ids.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(bool).to(input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).long().to(input_ids.device)

        bert_output = self.language_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        language_feature = self.proj_to_caption_model(bert_output.last_hidden_state)
        language_feature = language_feature + self.pos_emb(seq)

        # fine tuning the pretrained BERT-based model
        language_feature = self.encoder_layer(queries=language_feature,
                                                keys=language_feature, 
                                                values=language_feature,
                                                padding_mask=mask_queries.unsqueeze(-1),
                                                attention_mask=mask_self_attention)

        logits = self.proj_to_vocab(language_feature)
        out = F.log_softmax(logits, dim=-1)
        return out, language_feature

Pretrained_language_models = {
    "bert-base": BERTModel,
    "bert-large": BERTModel,
    "phobert-base": PhoBERTModel,
    "phobert-large": PhoBERTModel
}

def get_pretrained_language_model(model: str):
    return Pretrained_language_models[model]

def get_language_model(vocab, config):
    language_model = Pretrained_language_models[config.model.transformer.decoder.args.pretrained_language_model]
    return language_model(vocab, config.model.transformer.decoder.args.pretrained_language_model_name,
                            language_model_hidden_size=config.model.transformer.decoder.args.language_model_hidden_size,
                            d_model=config.model.d_model, d_k=config.model.d_k, d_v=config.model.d_v, h=config.model.nhead,
                            d_ff=config.model.d_ff, max_len=vocab.max_caption_length, dropout=config.model.dropout)
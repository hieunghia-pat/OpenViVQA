import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.modules.attentions import AdaptiveScaledDotProductAttention, MultiHeadAttention, ScaledDotProductAttention
from models.utils import generate_padding_mask, generate_sequential_mask, sinusoid_encoding_table
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.visual_embeddings import Embedding, SinusoidPositionalEmbedding
from models.modules.containers import Module, ModuleList

import os

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 use_aoa=False, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        if self_att_module is None:
            self_att_module = ScaledDotProductAttention
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=self_att_module,
                                            attention_module_kwargs=self_att_module_kwargs)

        if enc_att_module is None:
            enc_att_module = ScaledDotProductAttention
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=enc_att_module,
                                            attention_module_kwargs=enc_att_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, padding_mask=None, self_attention_mask=None, enc_attention_mask=None):
        self_att = self.self_attn(queries, queries, queries, attention_mask=self_attention_mask)
        enc_att = self.enc_attn(self_att, keys, values, attention_mask=enc_attention_mask)

        padding_mask = padding_mask.view(enc_att.shape[:2]).unsqueeze(-1)  # (bs, seq_len, 1)
        enc_att = enc_att.masked_fill(padding_mask, value=0)

        ff = self.pwff(enc_att)
        
        return ff

class MeshedDecoderLayer(Module):
    def __init__(self, N_enc, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 use_aoa=False, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=self_att_module,
                                            attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=enc_att_module,
                                            attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.N_enc = N_enc
        self.fc_alphas = nn.ModuleList([nn.Linear(d_model + d_model, d_model) for _ in range(N_enc)])

        self.init_weights()

    def init_weights(self):
        for fc_alpha in self.fc_alphas:
            nn.init.xavier_uniform_(fc_alpha.weight)

        for fc_alpha in self.fc_alphas:
            nn.init.constant_(fc_alpha.bias, 0)

    def forward(self, queries, keys, values, padding_mask=None, self_attention_mask=None, enc_attention_mask=None):    
        self_att = self.self_att(queries, queries, queries, attention_mask=self_attention_mask)
        padding_mask = padding_mask.view(enc_att.shape[:2]).unsqueeze(-1)  # (bs, seq_len, 1)

        enc_atts = []
        for ith in range(self.N_enc):
            enc_att = self.enc_att(self_att, keys[:, ith], values[:, ith], 
                            attention_mask=enc_attention_mask)
            enc_att = enc_att.masked_fill(padding_mask, value=0)
            enc_atts.append(enc_att)

        alphas = []
        for fc_alpha, enc_att in zip(self.fc_alphas, enc_atts):
            alphas.append(torch.sigmoid(fc_alpha(torch.cat([self_att, enc_att], -1))))

        attn = 0
        for alpha, enc_att in zip(alphas, enc_atts):
            attn += enc_att * alpha
        attn = attn / np.sqrt(self.N_enc)

        ff = self.pwff(attn)
        ff = ff.masked_fill(padding_mask, value=0)

        return ff

class AdaptiveDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(AdaptiveDecoderLayer, self).__init__()
        if self_att_module is None:
            self_att_module = ScaledDotProductAttention
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        if enc_att_module is None:
            enc_att_module = AdaptiveScaledDotProductAttention
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False, 
                                            attention_module=enc_att_module, 
                                            attention_module_kwargs=enc_att_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, language_signals, padding_mask=None, self_attention_mask=None, enc_attention_mask=None):
        self_att = self.self_attn(queries, queries, queries, attention_mask=self_attention_mask)
        
        enc_att = self.enc_attn(self_att, keys, values, language_signals=language_signals, attention_mask=enc_attention_mask)
        padding_mask = padding_mask.view(enc_att.shape[:2]).unsqueeze(-1)  # (bs, seq_len, 1)
        enc_att = enc_att.masked_fill(padding_mask, value=0)

        ff = self.pwff(enc_att)

        return ff

class Decoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, weights=None,
                 use_aoa=False, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.word_emb = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.token_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.feature_pos_emb = SinusoidPositionalEmbedding(d_model, normalize=False)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, use_aoa=use_aoa,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, linguistics, fused_outputs):
        answer_tokens = linguistics.answer_tokens
        b_s, seq_len = answer_tokens.shape[:2]
        answer_padding_masks = generate_padding_mask(answer_tokens, self.padding_idx).to(answer_tokens.device)
        answer_self_attention_masks = generate_sequential_mask(seq_len).to(answer_tokens.device)
        answer_self_attention_masks = torch.logical_or(answer_padding_masks, answer_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, answer_self_attention_masks], -1)
            answer_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(answer_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(answer_padding_masks.squeeze(1).squeeze(1), 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        features = fused_outputs.features
        feature_pos_embeddings = self.feature_pos_emb(features)
        feature_attention_masks = fused_outputs.padding_masks

        # special process for the beam search of inference
        if features.shape[0] > feature_pos_embeddings.shape[0]:
            assert features.shape[0] % feature_pos_embeddings.shape[0] == 0
            beam_size = int(features.shape[0] / feature_pos_embeddings.shape[0])
            feature_pos_embeddings = feature_pos_embeddings.unsqueeze(1)  # (bs, 1, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.expand(feature_pos_embeddings.shape[0], feature_pos_embeddings.shape[1]*beam_size, 
                                                    feature_pos_embeddings.shape[2], feature_pos_embeddings.shape[3])  # (bs, beam_size, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.contiguous().flatten(0, 1)  # (bs*beam_size, seq_len, d_model)

        out = self.word_emb(answer_tokens)
        features = features + feature_pos_embeddings
        for layer in self.layers:
            out = out + self.token_pos_emb(seq)
            out = layer(out, features, features, padding_mask=answer_padding_masks,
                        self_attention_mask=answer_self_attention_masks, enc_attention_mask=feature_attention_masks)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_enc, N_dec, padding_idx, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, weights=None,
                 use_aoa=False, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.token_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.feature_pos_emb = SinusoidPositionalEmbedding(d_model, normalize=False)
        self.layers = ModuleList(
            [MeshedDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, use_aoa=use_aoa,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, linguistics, fused_outputs):

        answer_tokens = linguistics.answer_tokens
        b_s, seq_len = answer_tokens.shape[:2]
        answer_padding_masks = generate_padding_mask(answer_tokens, self.padding_idx).to(answer_tokens.device)
        answer_self_attention_masks = generate_sequential_mask(seq_len).to(answer_tokens.device)
        answer_self_attention_masks = torch.logical_or(answer_padding_masks, answer_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, answer_self_attention_masks], -1)
            answer_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(answer_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(answer_padding_masks, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        features = fused_outputs.features
        feature_pos_embeddings = self.feature_pos_emb(features)
        feature_attention_masks = fused_outputs.padding_masks

        # special process for the beam search of inference
        if features.shape[0] > feature_pos_embeddings.shape[0]:
            assert features.shape[0] % feature_pos_embeddings.shape[0] == 0
            beam_size = int(features.shape[0] / feature_pos_embeddings.shape[0])
            feature_pos_embeddings = feature_pos_embeddings.unsqueeze(1)  # (bs, 1, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.expand(feature_pos_embeddings.shape[0], feature_pos_embeddings.shape[1]*beam_size, 
                                                    feature_pos_embeddings.shape[2], feature_pos_embeddings.shape[3])  # (bs, beam_size, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.contiguous().flatten(0, 1)  # (bs*beam_size, seq_len, d_model)

        out = self.word_emb(answer_tokens)
        features = features + feature_pos_embeddings
        for layer in self.layers:
            out = out + self.token_pos_emb(seq)
            out = layer(out, features, features, padding_mask=answer_padding_masks,
                        self_attention_mask=answer_self_attention_masks, enc_attention_mask=feature_attention_masks)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class AdaptiveDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, pretrained_language_model_name, checkpoint_path,
                    pretrained_language_model, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048,
                    language_model_hidden_size=768, dropout=.1, weights=None, 
                    self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(AdaptiveDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.token_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.feature_pos_emb = SinusoidPositionalEmbedding(d_model, normalize=False)
        self.layers = ModuleList(
            [   DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, 
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, 
                                enc_att_module_kwargs=enc_att_module_kwargs) 
            if i < N_dec else 
                AdaptiveDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                        self_att_module=self_att_module, enc_att_module=enc_att_module, 
                                        self_att_module_kwargs=self_att_module_kwargs, 
                                        enc_att_module_kwargs=enc_att_module_kwargs) for i in range(N_dec + 1)
            ]
        )
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        # load and froze the language model
        self.language_model = pretrained_language_model(padding_idx=padding_idx, language_model_hidden_size=language_model_hidden_size, 
                                            pretrained_language_model_name=pretrained_language_model_name,
                                            vocab_size=vocab_size, max_len=max_len)
        
        language_model_path = os.path.join(checkpoint_path, f"{pretrained_language_model_name}.pth")
        # BERT-based model has been pretrained
        if os.path.isfile(language_model_path):
            model_file = torch.load(language_model_path)
            self.language_model.load_state_dict(model_file['state_dict'], strict=False)
        else: # fine tuning the BERT-based model in end-to-end way
            for p in self.language_model.parameters():
                p.requires_grad = False

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, linguistics, fused_outputs):

        answer_tokens = linguistics.answer_tokens
        b_s, seq_len = answer_tokens.shape[:2]
        answer_padding_masks = generate_padding_mask(answer_tokens, self.padding_idx).to(answer_tokens.device)
        answer_self_attention_masks = generate_sequential_mask(seq_len).to(answer_tokens.device)
        answer_self_attention_masks = torch.logical_or(answer_padding_masks, answer_self_attention_masks)
        
        b_s, seq_len = answer_tokens.shape[:2]
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, answer_self_attention_masks], -1)
            answer_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(answer_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(answer_padding_masks, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        features = fused_outputs.features
        feature_pos_embeddings = self.feature_pos_emb(features)
        feature_attention_masks = fused_outputs.padding_masks

        # special process for the beam search of inference
        if features.shape[0] > feature_pos_embeddings.shape[0]:
            assert features.shape[0] % feature_pos_embeddings.shape[0] == 0
            beam_size = int(features.shape[0] / feature_pos_embeddings.shape[0])
            feature_pos_embeddings = feature_pos_embeddings.unsqueeze(1)  # (bs, 1, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.expand(feature_pos_embeddings.shape[0], feature_pos_embeddings.shape[1]*beam_size, 
                                                    feature_pos_embeddings.shape[2], feature_pos_embeddings.shape[3])  # (bs, beam_size, seq_len, d_model)
            feature_pos_embeddings = feature_pos_embeddings.contiguous().flatten(0, 1)  # (bs*beam_size, seq_len, d_model)

        # get the language_signals
        _, language_signals = self.language_model(answer_tokens)

        out = self.word_emb(answer_tokens) + self.token_pos_emb(seq)
        features = features + feature_pos_embeddings
        for i, layer in enumerate(self.layers):
            out = out + self.pos_emb(seq)
            if i < self.N:
                out = layer(out, features, features, padding_mask=answer_padding_masks,
                            self_attention_mask=answer_self_attention_masks, enc_attention_mask=feature_attention_masks)
            else:
                out = layer(out, features, features, language_signals=language_signals, padding_mask=answer_padding_masks,
                            self_attention_mask=answer_self_attention_masks, enc_attention_mask=feature_attention_masks)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

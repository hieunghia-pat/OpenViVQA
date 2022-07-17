import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

from models.language_models import get_pretrained_language_model
from models.modules.attentions import AdaptiveScaledDotProductAttention, MultiHeadAttention, ScaledDotProductAttention
from models.utils import generate_sequential_mask, sinusoid_encoding_table, generate_padding_mask
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.embeddings import Embedding
from models.modules.containers import Module, ModuleList

class DecoderLayer(Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, use_aoa=False, **attention_module_kwargs):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, padding_mask, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries=queries, keys=queries, values=queries, attention_mask=self_attention_mask, **kwargs)
        self_att = self_att.masked_fill(padding_mask, value=0)

        enc_att = self.enc_attn(queries=self_att, keys=keys, values=values, attention_mask=enc_attention_mask, **kwargs)
        enc_att = enc_att.masked_fill(padding_mask, value=0)

        ff = self.pwff(enc_att)
        
        return ff

class MeshedDecoderLayer(Module):
    def __init__(self, N_enc=3, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, use_aoa=False, **attention_module_kwargs):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa,
                                            attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                            use_aoa=use_aoa,
                                            attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)
        
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.N_enc = N_enc
        self.fc_alphas = nn.ModuleList([nn.Linear(d_model + d_model, d_model) for _ in range(N_enc)])

        self.init_weights()

    def init_weights(self):
        for fc_alpha in self.fc_alphas:
            nn.init.xavier_uniform_(fc_alpha.weight)

        for fc_alpha in self.fc_alphas:
            nn.init.constant_(fc_alpha.bias, 0)

    def forward(self, queries, keys, values, padding_mask, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_att(queries=queries, keys=queries, values=queries, attention_mask=self_attention_mask, **kwargs)
        self_att = self_att.masked_fill(padding_mask, value=0)

        enc_atts = []
        for ith in range(self.N_enc):
            enc_att = self.enc_att(queries=self_att, keys=keys[:, ith], values=values[:, ith], 
                            attention_mask=enc_attention_mask, **kwargs)
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
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, use_aoa=False, **attention_module_kwargs):
        super(AdaptiveDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa, attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False, 
                                            use_aoa=use_aoa, attention_module=AdaptiveScaledDotProductAttention, 
                                            **attention_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, language_signals, padding_mask=None, self_attention_mask=None, enc_attention_mask=None, **kwargs):
        self_att = self.self_attn(queries=queries, keys=queries, values=queries, attention_mask=self_attention_mask, **kwargs)
        self_att = self_att.masked_fill(padding_mask, value=0)

        enc_att = self.enc_attn(queries=self_att, keys=keys, values=values, language_signals=language_signals, attention_mask=enc_attention_mask, **kwargs)
        enc_att = enc_att.masked_fill(padding_mask, value=0)

        ff = self.pwff(enc_att)
        ff = ff.masked_fill(padding_mask, value=0)

        return ff

class MeshedAdaptiveDecoderLayer(Module):
    def __init__(self, N_enc=3, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, use_aoa=False, **attention_module_kwargs):
        super(MeshedAdaptiveDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                            use_aoa=use_aoa, attention_module=ScaledDotProductAttention,
                                            **attention_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False, 
                                            use_aoa=use_aoa, attention_module=AdaptiveScaledDotProductAttention, 
                                            **attention_module_kwargs)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.N_enc = N_enc
        self.fc_alphas = nn.ModuleList([nn.Linear(d_model + d_model, d_model) for _ in range(N_enc)])

        self.init_weights()

    def init_weights(self):
        for fc_alpha in self.fc_alphas:
            nn.init.xavier_uniform_(fc_alpha.weight)

        for fc_alpha in self.fc_alphas:
            nn.init.constant_(fc_alpha.bias, 0)

    def forward(self, queries, keys, values, language_signals, padding_mask=None, self_attention_mask=None, enc_attention_mask=None, **kwargs):
        self_att = self.self_att(queries=queries, keys=queries, values=queries, attention_mask=self_attention_mask)
        self_att = self_att.masked_fill(padding_mask, value=0)

        enc_atts = []
        for ith in range(self.N_enc):
            enc_att = self.enc_att(queries=self_att, keys=keys[:, ith], values=values[:, ith], 
                            language_signals=language_signals, attention_mask=enc_attention_mask)
            enc_att = enc_att.masked_fill(padding_mask, value=0)
            enc_atts.append(enc_att)

        alphas = []
        for fc_alpha, enc_att in zip(self.fc_alphas, enc_atts):
            alphas.append(torch.sigmoid(fc_alpha(torch.cat([self_att, enc_att], -1))))

        attn = 0
        for alpha, enc_att in zip(alphas, enc_atts):
            attn += enc_att * alpha
        attn = attn / np.sqrt(self.N_enc)
        attn = attn.masked_fill(padding_mask, value=0)

        ff = self.pwff(attn)
        ff = ff.masked_fill(padding_mask, value=0)

        return ff

class Decoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, vocab, max_len, N_dec, padding_idx, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, weights=None,
                 use_aoa=False, **attention_module_kwargs):
        super(Decoder, self).__init__()
        vocab_size = len(vocab)

        self.d_model = d_model
        self.word_embedding = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, tokens, enc_outputs, enc_attention_mask, **kwargs):
        b_s, seq_len = tokens.shape[:2]
        mask_queries = generate_padding_mask(tokens, self.padding_idx).to(tokens.device)  # (b_s, seq_len)
        self_attention_mask = generate_sequential_mask(seq_len).to(tokens.device)
        self_attention_mask = self_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        self_attention_mask = torch.logical_or(self_attention_mask, mask_queries.unsqueeze(1).unsqueeze(1))
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, self_attention_mask], -1)
            self_attention_mask = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_embedding(tokens) + self.pos_embedding(seq)

        for layer in self.layers:
            out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                        padding_mask=mask_queries.unsqueeze(-1), 
                        self_attention_mask=self_attention_mask, 
                        enc_attention_mask=enc_attention_mask)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class MeshedDecoder(Module):
    def __init__(self, vocab, max_len, N_enc, N_dec, padding_idx, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, weights=None,
                 use_aoa=False, **attention_module_kwargs):
        super(MeshedDecoder, self).__init__()
        vocab_size = len(vocab)

        self.d_model = d_model
        self.word_embedding = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, tokens, enc_outputs, enc_pos_embedding, enc_attention_mask, **kwargs):
        enc_outputs += enc_pos_embedding

        b_s, seq_len = tokens.shape[:2]
        mask_queries = generate_padding_mask(tokens, self.padding_idx).to(tokens.device)  # (b_s, seq_len)
        self_attention_mask = generate_sequential_mask(seq_len).to(tokens.device)
        self_attention_mask = self_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        self_attention_mask = torch.logical_or(self_attention_mask, mask_queries.unsqueeze(1).unsqueeze(1))
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, self_attention_mask], -1)
            self_attention_mask = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_embedding(tokens)
        word_pos_embedding = self.pos_embedding(seq)

        for layer in self.layers:
            out += word_pos_embedding
            out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                        padding_mask=mask_queries.unsqueeze(-1), 
                        self_attention_mask=self_attention_mask, 
                        enc_attention_mask=enc_attention_mask)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class AdaptiveDecoder(Module):
    def __init__(self, vocab, max_len, N_dec, padding_idx, pretrained_language_model_name, 
                    pretrained_language_model, pretrained_language_model_path=None, use_aoa=False, d_model=512, d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048,
                    language_model_hidden_size=768, dropout=.1, weights=None, **attention_module_kwargs):
        super(AdaptiveDecoder, self).__init__()
        vocab_size = len(vocab)

        self.d_model = d_model
        self.word_embbeding = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.pos_embbeding = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [   DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs) 
            if i < N_dec else 
                AdaptiveDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs) for i in range(N_dec + 1)
            ]
        )
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        # load and froze the language model
        language_model = get_pretrained_language_model(pretrained_language_model)
        self.language_model = language_model(vocab, pretrained_language_model_name, d_model=d_model,
                                            language_model_hidden_size=language_model_hidden_size,
                                            d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, max_len=max_len, dropout=dropout)

        if os.path.isfile(pretrained_language_model_path):
            print("In AdaptiveDecoder: Loading the pretrained language model ..")
            language_model_checkpoint = torch.load(pretrained_language_model_path)
            self.language_model.load_state_dict(language_model_checkpoint["state_dict"])
            # frozen the language model
            for param in self.language_model.parameters():
                param.requires_grad = False
        else:
            print("In AdaptiveDecoder: Fine-tuning the language model while training the captioning model")

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, tokens, enc_outputs, enc_pos_embedding, enc_attention_mask, **kwargs):
        enc_outputs += enc_pos_embedding

        b_s, seq_len = tokens.shape[:2]
        mask_queries = generate_padding_mask(tokens, self.padding_idx).to(tokens.device)  # (b_s, seq_len)
        self_attention_mask = generate_sequential_mask(seq_len).to(tokens.device)
        self_attention_mask = self_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        self_attention_mask = torch.logical_or(self_attention_mask, mask_queries.unsqueeze(1).unsqueeze(1))
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, self_attention_mask], -1)
            self_attention_mask = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_embbeding(tokens)
        word_pos_embedding = self.pos_embbeding(seq)

        _, language_feature = self.language_model(tokens, attention_mask=torch.logical_not(mask_queries))

        for i, layer in enumerate(self.layers):
            out += word_pos_embedding
            if i < self.N:
                out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                            padding_mask=mask_queries.unsqueeze(-1), 
                            self_attention_mask=self_attention_mask, 
                            enc_attention_mask=enc_attention_mask)
            else:
                out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                            padding_mask=mask_queries.unsqueeze(-1),
                            language_signals=language_feature, 
                            self_attention_mask=self_attention_mask, 
                            enc_attention_mask=enc_attention_mask)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

class MeshedAdaptiveDecoder(Module):
    def __init__(self, vocab, max_len, N_dec, padding_idx, pretrained_language_model_name, 
                    pretrained_language_model, pretrained_language_model_path: str=None, d_model=512, 
                    d_emb=None, d_k=64, d_v=64, h=8, d_ff=2048,
                    language_model_hidden_size=768, dropout=.1, weights=None,
                 use_aoa=False, N_enc=3, **attention_module_kwargs):
        
        super(MeshedAdaptiveDecoder, self).__init__()
        vocab_size = len(vocab)

        self.d_model = d_model
        self.word_embbeding = Embedding(vocab_size, d_model=d_model, d_emb=d_emb, weights=weights, padding_idx=padding_idx)
        self.pos_embbeding = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [   MeshedDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs)
            if i < N_dec else 
                MeshedAdaptiveDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, use_aoa=use_aoa, **attention_module_kwargs) for i in range(N_dec + 1)
            ]
        )

        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        # load and froze the language model
        language_model = get_pretrained_language_model(pretrained_language_model)
        self.language_model = language_model(vocab, pretrained_language_model_name, d_model=d_model,
                                            language_model_hidden_size=language_model_hidden_size,
                                            d_k=d_k, d_v=d_v, h=h, d_ff=d_ff, max_len=max_len, dropout=dropout)

        if os.path.isfile(pretrained_language_model_path):
            print("In AdaptiveDecoder: Loading the pretrained language model ..")
            language_model_checkpoint = torch.load(pretrained_language_model_path)
            self.language_model.load_state_dict(language_model_checkpoint["state_dict"])
            # frozen the language model
            for param in self.language_model.parameters():
                param.requires_grad = False
        else:
            print("In AdaptiveDecoder: Fine-tuning the language model while training the captioning model")

        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, tokens, enc_outputs, enc_pos_embedding, enc_attention_mask, **kwargs):
        enc_outputs += enc_pos_embedding

        b_s, seq_len = tokens.shape[:2]
        mask_queries = generate_padding_mask(tokens, self.padding_idx).to(tokens.device)  # (b_s, seq_len)
        self_attention_mask = generate_sequential_mask(seq_len).to(tokens.device)
        self_attention_mask = self_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        self_attention_mask = torch.logical_or(self_attention_mask, mask_queries.unsqueeze(1).unsqueeze(1))
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, self_attention_mask], -1)
            self_attention_mask = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_embbeding(tokens)
        word_pos_embedding = self.pos_embbeding(seq)
        _, language_feature = self.language_model(input, attention_mask = torch.logical_not(mask_queries))

        for i, layer in enumerate(self.layers):
            out += word_pos_embedding
            if i < self.N:
                out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                            padding_mask=mask_queries.unsqueeze(-1), 
                            self_attention_mask=self_attention_mask, 
                            enc_attention_mask=enc_attention_mask)
            else:
                out = layer(queries=out, keys=enc_outputs, values=enc_outputs,
                            padding_mask=mask_queries.unsqueeze(-1),
                            language_signals=language_feature, 
                            self_attention_mask=self_attention_mask, 
                            enc_attention_mask=enc_attention_mask)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

Decoders = {
    "decoder": Decoder,
    "meshed-decoder": MeshedDecoder,
    "adaptive-decoder": AdaptiveDecoder,
    "meshed-adaptive-decoder": MeshedAdaptiveDecoder
}

def get_decoder(vocab, config):
    decoder = Decoders[config.model.transformer.decoder.module]
    checkpoint_path = config.training.checkpoint_path
    language_model_name = config.model.transformer.decoder.args.pretrained_language_model
    pretrained_language_model_path = config.model.transformer.decoder.args.pretrained_language_model_path
    if language_model_name is not None:
        config.model.transformer.decoder.args.pretrained_language_model_path = os.path.join(checkpoint_path, 
                                                                                            language_model_name,
                                                                                            pretrained_language_model_path)

    return decoder(vocab=vocab, max_len=vocab.max_caption_length, N_dec=config.model.nlayers, 
                    padding_idx=vocab.padding_idx, d_model=config.model.d_model, d_k=config.model.d_k,
                    d_v=config.model.d_v, d_ff=config.model.d_ff, dropout=config.model.dropout,
                    **config.model.transformer.decoder.args)
import torch
from torch import nn

from data_utils.vocab import Vocab
from models.utils import generate_sequential_mask, generate_padding_mask
from models.language_models import Pretrained_language_models
from models.modules.encoders import Encoder

import math
from transformers import AutoModel

class Embedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model, embedding_dim, dropout=0.5):
        super(Embedding, self).__init__()

        self.padding_idx = vocab.padding_idx

        if vocab.vectors is None:
            self.embedding = nn.Embedding(len(vocab), embedding_dim, vocab.padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings=vocab.vectors, freeze=True, padding_idx=vocab.padding_idx)

        self.proj = nn.Linear(embedding_dim, d_model),
        self.activation = nn.GELU(),
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.embedding(tokens)

        return features, (padding_masks, sequential_masks)

class VisualEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(2048, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual):
        masks = generate_padding_mask(visual, padding_idx=0).to(visual.device)

        visual = self.activation(self.proj(visual))
        visual = self.dropout(visual)

        return visual, masks

class LSTMTextEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim, d_model, dropout=0.5):
        super(LSTMTextEmbedding, self).__init__()

        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.padding_idx)
        self.padding_idx = vocab.padding_idx
        if vocab.vectors is not None:
            self.embedding.from_pretrained(vocab.vectors)
        self.proj = nn.Linear(embedding_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.activation(self.proj(self.embedding(tokens)))
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)

class PretrainedLanguageModelEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model, pretrained_language_model, language_model_hidden_size, dropout=0.5):
        super(PretrainedLanguageModelEmbedding, self).__init__()

        self.embedding = AutoModel.from_pretrained(Pretrained_language_models[pretrained_language_model])
        self.padding_idx = vocab.padding_idx
        self.proj = nn.Linear(language_model_hidden_size, d_model)
        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)
        
        embedded_tokens = self.embedding(tokens)
        features = self.activation(self.proj(embedded_tokens))
        features = self.dropout(features)

        return features, (padding_masks, sequential_masks)

class TransferEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, pretrained_language_model, language_model_hidden_size, embedding_dim, d_model=512,
                    nlayers=1, d_k=64, d_q=64, d_v=64, h=8, d_ff=2048, dropout=0.5):
        super(TransferEmbedding, self).__init__()

        self.padding_idx = vocab.padding_idx
        
        if vocab.vectors is None:
            self.embedding = nn.Embedding(len(vocab), d_model, vocab.padding_idx)
        else:
            self.embedding = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=vocab.vectors, freeze=True, padding_idx=vocab.padding_idx),
                nn.Linear(embedding_dim, d_model),
                nn.GELU(),
                nn.Dropout(p=dropout)
            )

        # mapping from vocab space of dataset to vocab space of pretrained language model
        self.vocab_encoder = Encoder(N=nlayers, padding_idx=vocab.padding_idx, d_model=d_model,
                                        d_q=d_q, d_k=d_k, d_v=d_v, h=h, d_ff=d_ff)

        self.proj_to_pretrained = nn.Linear(embedding_dim, language_model_hidden_size)
        self.activation_1 = nn.GELU()
        self.dropout_1 = nn.Dropout(p=dropout)

        self.language_model = AutoModel.from_pretrained(pretrained_language_model)

        self.proj_to_model = nn.Linear(language_model_hidden_size, d_model)
        self.activation_2 = nn.GELU()
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        embedded_tokens = self.embedding(tokens)
        features = self.vocab_encoder(embedded_tokens)

        features = self.dropout_1(self.activation_1(self.proj_to_pretrained(features)))
        features = self.language_model(features)

        features = self.dropout_2(self.activation_2(self.proj_to_model(features)))

        return features, (padding_masks, sequential_masks)

class SinusoidPositionalEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # returns an 1D embedding for visual feature
    # follow implementation of https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/position_encoding.py

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(SinusoidPositionalEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=-1).flatten(-2)

        return pos

Embeddings = {
    "standard-embedding": Embedding,
    "visual-embedding": VisualEmbedding,
    "lstm-embedding": LSTMTextEmbedding,
    "pretrained-language-model-embedding": PretrainedLanguageModelEmbedding,
    "transfer-embedding": TransferEmbedding
}

def get_visual_embedding(vocab, config):
    embedding_module = Embeddings[config.model.visual_embedding.module]
    return embedding_module(vocab=vocab, d_model=config.model.d_model, **config.visual_embedding.args)

def get_linguistic_embedding(vocab, config):
    embedding_module = Embeddings[config.model.linguistic_embedding.module]
    return embedding_module(vocab=vocab, d_model=config.model.d_model, **config.linguistic_embedding.args)
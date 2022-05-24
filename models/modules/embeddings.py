import torch
from torch import nn
from data_utils.vocab import Vocab

from models.utils import generate_sequential_mask, generate_padding_mask

import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, d_emb=None, weights=None, padding_idx=0):
        super(Embedding, self).__init__()
        if weights is None:
            self.components = nn.Embedding(vocab_size, d_model, padding_idx)
        else:
            assert d_emb != None, "d_emb must be specified when using pretrained word-embedding"
            self.components = nn.Sequential(
                nn.Linear(d_emb, d_model),
                nn.Embedding.from_pretrained(embeddings=weights, freeze=True, padding_idx=padding_idx)
            )

    def forward(self, tokens):
        return self.components(tokens)

class VisualEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual):
        masks = generate_padding_mask(visual, padding_idx=0).to(visual.device)

        visual = self.proj(visual)
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
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.proj(self.embedding(tokens))
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LearnedPositionalEmbedding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnedPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional embedding once in log space.
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)

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
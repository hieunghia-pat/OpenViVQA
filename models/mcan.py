import torch
from torch import nn

import numpy as np

from models.embedding import TextEmbedding, VisualEmbedding
from models.deep_co_attention import DeepCoAttention
from models.fusion_model import FusionModel

class MCAN(nn.Module):
    def __init__(self, vocab, backbone, d_model, embedding_dim, dff, nheads, nlayers, dropout):
        super(MCAN, self).__init__()

        self.padding_idx = vocab.stoi["<pad>"]

        # image and question representation
        self.visual_embedding = VisualEmbedding(d_model, dropout)
        self.text_embedding = TextEmbedding(vocab, embedding_dim, d_model, dropout)

        # deep co-attention learning
        self.deep_co_attention = DeepCoAttention(d_model, dff, nheads, nlayers, dropout)

        # fusion model
        self.fusion = FusionModel(d_model, dropout)

        # output classifier
        self.generator = nn.Linear(d_model, len(vocab.output_cats))
        self.dropout = nn.Dropout(dropout)

    def key_padding_mask(self, x, padding_idx):
        "Mask out subsequent positions."
        return x == padding_idx

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        return nn.Transformer.generate_square_subsequent_mask(size)

    def forward(self, v, q):
        device = v.device
        v_embedded = self.visual_embedding(v)
        q_embedded = self.text_embedding(q)

        q_attn_mask = self.subsequent_mask(q_embedded.size(1)).to(device)
        q_key_padding_mask = self.key_padding_mask(q, self.padding_idx).to(device)

        v_encoded, q_encoded = self.deep_co_attention(v_embedded, q_embedded, q_attn_mask, q_key_padding_mask)
        
        fused_features = self.fusion(v_encoded, q_encoded)

        outs = self.generator(fused_features)

        return self.dropout(outs)
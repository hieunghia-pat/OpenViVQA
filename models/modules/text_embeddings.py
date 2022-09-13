from torch import nn
from data_utils.vocab import Vocab

from builders.build_text_embedding import META_TEXT_EMBEDDING
from models.utils import generate_sequential_mask, generate_padding_mask

@META_TEXT_EMBEDDING.register()
class UsualEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, d_emb=None, weights=None, padding_idx=0):
        super(UsualEmbedding, self).__init__()
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

@META_TEXT_EMBEDDING.register()
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
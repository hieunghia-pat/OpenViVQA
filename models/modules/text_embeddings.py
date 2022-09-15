from torch import nn

from data_utils.vocab import Vocab
from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

@META_TEXT_EMBEDDING.register()
class UsualEmbedding(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(UsualEmbedding, self).__init__()
        if config.WORD_EMBEDDING is None:
            self.components = nn.Embedding(len(vocab), config.D_MODEL, vocab.padding_idx)
        else:
            embedding_weights = build_word_embedding(config).vectors
            self.components = nn.Sequential(
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.Dropout(config.DROPOUT),
                nn.Embedding.from_pretrained(embeddings=embedding_weights, freeze=True, padding_idx=vocab.padding_idx)
            )

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.components(tokens)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class LSTMTextEmbedding(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTMTextEmbedding, self).__init__()

        self.embedding = nn.Embedding(len(vocab), config.D_EMBEDDING, padding_idx=vocab.padding_idx)
        self.padding_idx = vocab.padding_idx
        if config.WORD_EMBEDDING is not None:
            embedding_weights = build_word_embedding(config).vectors
            self.embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=vocab.padding_idx)
        self.proj = nn.Linear(config.D_EMBEDDING, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.lstm = nn.LSTM(input_size=config.D_MODEL, hidden_size=config.D_MODEL, batch_first=True)

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.proj(self.embedding(tokens))
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)
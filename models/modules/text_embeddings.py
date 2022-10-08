import torch
from torch import nn

from data_utils.vocab import Vocab
from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers import AutoTokenizer, AutoModel

from typing import Dict, List

@META_TEXT_EMBEDDING.register()
class UsualEmbedding(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(UsualEmbedding, self).__init__()

        self.padding_idx = vocab.padding_idx

        if config.WORD_EMBEDDING is None:
            self.components = nn.Embedding(len(vocab), config.D_MODEL, vocab.padding_idx)
        else:
            embedding_weights = build_word_embedding(config).word_embeddings
            self.components = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=embedding_weights, freeze=True, padding_idx=vocab.padding_idx),
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.Dropout(config.DROPOUT)
            )

    def forward(self, tokens: List[str]):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.components(tokens)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class DynamicEmbedding(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__(config, vocab)

        self.padding_idx = vocab.padding_idx
        self.d_embedding = config.D_EMBEDDING
        self.len_vocab = len(vocab)
        self.register_parameter("weights", nn.parameter.Parameter(nn.init.xavier_normal_(torch.zeros((self.len_vocab, self.d_embedding)))))
        self.fc = nn.Linear(config.D_EMBEDDING, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.init_weights()

        if config.WORD_EMBEDDING is None:
            nn.init.xavier_uniform_(self.weights)
        else:
            self.weights = build_word_embedding(config).word_embeddings

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, tokens: torch.Tensor, maps_oov_to_features: List[Dict[int, torch.Tensor]]):
        # create one-hot vector
        bs, seq_len = tokens.shape
        len_vocab = max(self.len_vocab, tokens.max())
        one_hot_vecs = torch.zeros((bs, seq_len, len_vocab)).to(tokens.device)
        one_vecs = torch.zeros((bs, seq_len, len_vocab)).to(tokens.device)
        one_hot_vecs.scatter_(dim=-1, index=tokens.unsqueeze(-1), src=one_vecs)

@META_TEXT_EMBEDDING.register()
class OcrUsualEmbedding(UsualEmbedding):
    def __init__(self, config, vocab):
        super().__init__(self, config, vocab)

    def forward(self, tokens: List[List[str]]):
        list_of_features = []
        max_len = 0
        for tokens_per_batch in tokens:
            if len(tokens_per_batch) > 0:
                refined_tokens = []
                for token in tokens_per_batch:
                    refined_tokens += token.split()
                input_ids = self.vocab.encode_question(refined_tokens)
                features_per_batch = self.embedding(input_ids).to(self.device)
                features_per_batch = self.components(input_ids)
                features_per_batch = features_per_batch.sum(dim=1)
            else:
                features_per_batch = torch.ones((1, self.d_pretrained_feature)).to(self.embedding.device)
            list_of_features.append(features_per_batch)
            max_len = len(tokens_per_batch) if len(tokens_per_batch) > max_len else max_len
        padding_tensor = torch.zeros((1, self.d_pretrained_feature)).to(self.embedding.device)
        for idx, feature in enumerate(list_of_features):
            delta_len = max_len - feature.shape[0]
            list_of_features[idx] = torch.cat([feature] + [padding_tensor]*delta_len, dim=0)
        
        features = torch.cat([feature.unsqueeze(0) for feature in list_of_features], dim=0)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        sequential_mask = generate_sequential_mask(features.shape[1])

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, (padding_mask, sequential_mask)

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

    def forward(self, tokens: List[str]):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.proj(self.embedding(tokens)) # (bs, seq_len, d_model)
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class OcrLSTMEmbedding(LSTMTextEmbedding):
    def __init__(self, config, vocab):
        super().__init__(self, config, vocab)

        self.vocab = vocab

    def forward(self, tokens: List[List[str]]):
        list_of_features = []
        max_len = 0
        for tokens_per_batch in tokens:
            if len(tokens_per_batch) > 0:
                refined_tokens = []
                for token in tokens_per_batch:
                    refined_tokens += token.split()
                input_ids = self.vocab.encoder_question(refined_tokens)
                features_per_batch = self.embedding(input_ids).to(self.device)
                features_per_batch = self.dropout(self.proj(features_per_batch))
                features_per_batch, _ = self.lstm(features_per_batch)
                features_per_batch = features_per_batch.sum(dim=1)
            else:
                features_per_batch = torch.ones((1, self.d_pretrained_feature)).to(self.embedding.device)
            list_of_features.append(features_per_batch)
            max_len = len(tokens_per_batch) if len(tokens_per_batch) > max_len else max_len
        padding_tensor = torch.zeros((1, self.d_pretrained_feature)).to(self.embedding.device)
        for idx, feature in enumerate(list_of_features):
            delta_len = max_len - feature.shape[0]
            list_of_features[idx] = torch.cat([feature] + [padding_tensor]*delta_len, dim=0)
        
        features = torch.cat([feature.unsqueeze(0) for feature in list_of_features], dim=0)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        sequential_mask = generate_sequential_mask(features.shape[1])

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, (padding_mask, sequential_mask)

@META_TEXT_EMBEDDING.register()
class BertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = AutoModel.from_pretrained(config.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, tokens: List[str]):
        inputs = self.tokenizer(tokens, return_tensors="pt", padding=True).to(self.device)
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        sequential_mask = generate_sequential_mask(tokens.shape[1])
        features = self.embedding(**inputs).last_hidden_state

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, (padding_mask, sequential_mask)

@META_TEXT_EMBEDDING.register()
class OcrBertEmbedding(BertEmbedding):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.d_pretrained_feature = config.D_PRETRAINED_FEATURE

    def forward(self, tokens: List[List[str]]):
        list_of_features = []
        max_len = 0
        for tokens_per_batch in tokens:
            if len(tokens_per_batch) > 0:
                inputs = self.tokenizer(tokens_per_batch, return_tensors="pt", padding=True).to(self.device)
                features_per_batch = self.embedding(**inputs).last_hidden_state
                features_per_batch = features_per_batch.sum(dim=1)
            else:
                features_per_batch = torch.ones((1, self.d_pretrained_feature)).to(self.embedding.device)
            list_of_features.append(features_per_batch)
            max_len = len(tokens_per_batch) if len(tokens_per_batch) > max_len else max_len
        padding_tensor = torch.zeros((1, self.d_pretrained_feature)).to(self.embedding.device)
        for idx, feature in enumerate(list_of_features):
            delta_len = max_len - feature.shape[0]
            list_of_features[idx] = torch.cat([feature] + [padding_tensor]*delta_len, dim=0)
        
        features = torch.cat([feature.unsqueeze(0) for feature in list_of_features], dim=0)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        sequential_mask = generate_sequential_mask(features.shape[1])

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, (padding_mask, sequential_mask)

@META_TEXT_EMBEDDING.register()
class T5Embedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = AutoModel.from_pretrained(config.PRETRAINED_NAME)

    def forward(self, tokens: List[str]):
        input_ids = self.tokenizer(tokens, return_tensors='pt', padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)
        sequential_mask = generate_sequential_mask(input_ids.shape[1])

        out = self.embedding(input_ids=input_ids, decoder_input_ids=input_ids).last_hidden_states

        return out, (padding_mask, sequential_mask)
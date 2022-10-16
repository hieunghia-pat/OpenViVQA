import torch
from torch import nn
from torch.nn import functional as F

from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel, T5Tokenizer, T5EncoderModel

import os
import numpy as np
from typing import Dict, List, Union
import itertools
from collections import defaultdict
from copy import deepcopy

@META_TEXT_EMBEDDING.register()
class UsualEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super(UsualEmbedding, self).__init__()

        self.padding_idx = vocab.padding_idx

        if config.WORD_EMBEDDING is None:
            self.components = nn.Embedding(len(vocab), config.D_MODEL, vocab.padding_idx)
        else:
            embedding_weights = build_word_embedding(config).vectors
            self.components = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=embedding_weights, freeze=True, padding_idx=vocab.padding_idx),
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.Dropout(config.DROPOUT)
            )

    def forward(self, tokens):
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.components(tokens)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class OcrEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.padding_idx = 0
        self.device = config.DEVICE
        self.ocr_token = vocab.ocr_token

        ocr_texts = []
        for file in os.listdir(config.OCR_PATH):
            ocr_features = np.load(os.path.join(config.OCR_PATH, file), allow_pickle=True)[()]
            ocr_texts.extend(itertools.chain(*[text.split() for text in ocr_features["texts"]]))
        ocr_texts = set(ocr_texts)
        self.stoi = {token: i for i, token in enumerate(ocr_texts)}
        self.stoi.update({self.ocr_token: len(self.stoi)})
        self.itos = {i: token for token, i in self.stoi.items()}

        if config.WORD_EMBEDDING is None: # define the customized vocab
            self.components = nn.Embedding(len(self.itos), config.D_MODEL, self.padding_idx)
        else:
            self.load_word_embeddings(build_word_embedding(config))
            self.components = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=self.word_embeddings, freeze=True, padding_idx=self.padding_idx),
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.Dropout(config.DROPOUT)
            )
    
    def load_word_embeddings(self, word_embeddings):
        if not isinstance(word_embeddings, list):
            word_embeddings = [word_embeddings]

        tot_dim = sum(embedding.dim for embedding in word_embeddings)
        self.word_embeddings = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in word_embeddings:
                end_dim = start_dim + v.dim
                self.word_embeddings[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def forward(self, texts: List[List[str]]):
        max_len = max([len(text) for text in texts])
        for idx, text in enumerate(texts):
            if len(text) < max_len:
                text.extend([self.ocr_token] * (max_len-len(text)))
            texts[idx] = text
        '''
            features: List[List[torch.Tensor]] - a batch of list of embedded features,
                                in which each each features is the sum of embedded features of sub-tokens splitted from token
        '''
        features = deepcopy(texts)
        for batch, text in enumerate(texts):
            for idx, tokens in enumerate(text):
                token = [self.stoi[token] if token in self.stoi else self.stoi[self.ocr_token] for token in tokens.split()]
                token = torch.tensor(token).unsqueeze(0).to(self.device)
                feature = self.components(token).sum(dim=1)
                features[batch][idx] = feature
            features[batch] = torch.cat(features[batch], dim=0).unsqueeze(0)
        features = torch.cat(features, dim=0)

        return features, None

@META_TEXT_EMBEDDING.register()
class DynamicEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.d_model = config.D_MODEL
        self.vocab = vocab

        self.register_parameter("fixed_weights", nn.parameter.Parameter(nn.init.xavier_uniform_(torch.ones((len(vocab), self.d_model)))))

    def batch_embedding(self, weights, tokens, padding_idx):
        '''
            weights: (bs, embedding_len, d_model)
            tokens: (bs, seq_len)
        '''
        assert weights.dim() == 3
        batch_size = weights.shape[0]
        length = weights.shape[1]
        d_model = weights.shape[-1]
        assert d_model == self.d_model
        flattened_weights = weights.view(batch_size*length, d_model)

        batch_offsets = torch.arange(batch_size, device=tokens.device) * length
        batch_offsets = batch_offsets.unsqueeze(-1)
        assert batch_offsets.dim() == tokens.dim()
        flattened_tokens = tokens + batch_offsets
        results = F.embedding(flattened_tokens, flattened_weights, padding_idx=padding_idx)
        
        return results

    def forward(self, tokens: torch.Tensor, oov_features: torch.Tensor):
        padding_mask = generate_padding_mask(tokens, padding_idx=self.vocab.padding_idx).to(oov_features.device)
        seq_len = tokens.shape[1]
        sequential_mask = generate_sequential_mask(seq_len).to(oov_features.device)

        # construct the dynamic embeding weights
        bs = tokens.shape[0]
        fixed_weights = self.fixed_weights.unsqueeze(0).expand((bs, -1, -1)) # (bs, vocab_len, d_model)
        weights = torch.cat([fixed_weights, oov_features], dim=1) # (bs, vocab_len + ocr_len, d_model)

        features = self.batch_embedding(weights, tokens, self.vocab.padding_idx)

        return features, (padding_mask, sequential_mask)

@META_TEXT_EMBEDDING.register()
class LSTMTextEmbedding(nn.Module):
    def __init__(self, config, vocab):
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

        features = self.proj(self.embedding(tokens)) # (bs, seq_len, d_model)
        features = self.dropout(features)

        features, _ = self.lstm(features)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class BertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = BertModel.from_pretrained(config.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).to(self.device)
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(**inputs).last_hidden_state

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask

@META_TEXT_EMBEDDING.register()
class AlbertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = AlbertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = AlbertModel.from_pretrained(config.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).to(self.device)
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(**inputs).last_hidden_state

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask

@META_TEXT_EMBEDDING.register()
class T5Embedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = T5Tokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = T5EncoderModel.from_pretrained(config.PRETRAINED_NAME)

    def forward(self, questions: List[str]):
        input_ids = self.tokenizer(questions, return_tensors='pt', padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)

        out = self.embedding(input_ids=input_ids, decoder_input_ids=input_ids).last_hidden_states

        return out, padding_mask
import torch
from torch import nn
from torch.nn import functional as F

from data_utils.vocab import Vocab
from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel, T5Tokenizer, T5EncoderModel

import numpy as np
from typing import List
import itertools
from collections import defaultdict

@META_TEXT_EMBEDDING.register()
class UsualEmbedding(nn.Module):
    def __init__(self, config, vocab: Vocab):
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
class OcrDynamicEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.d_model = config.D_MODEL
        self.vocab = vocab

        if config.WORD_EMBEDDING is None:
            self.fixed_weights = nn.Embedding(len(vocab), config.D_MODEL, vocab.padding_idx)
            self.register_parameter("fixed_weights", nn.parameter.Parameter(nn.init.xavier_uniform_(torch.ones((len(vocab), self.d_model)))))
        else:
            embedding_weights = build_word_embedding(config).vectors
            self.register_parameter("fixed_weights", nn.parameter.Parameter(embedding_weights))

    def match_text_to_indices(self, text: List[str], vocab2idx, ocr2inds):
        """
            Match an text to a list of sequences of indices
            each index corresponds to either a fixed vocabulary or an OCR token
            (in the index address space, the OCR tokens are after the fixed vocab)
        """
        answer_word_matches = []
        for word in text:
            # match word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx:
                matched_inds.append(vocab2idx.get(word))
            # match answer word to OCR
            matched_inds.extend(ocr2inds[word])
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [
                seq + (idx,)
                for seq in idx_seq_list for idx in matched_inds
            ]

        return idx_seq_list

    def forward(self, list_of_texts: List[List[str]], ocr_tokens: List[List[str]], ocr_features: torch.Tensor):
        flattened_ocr_tokens = {idx: token for idx, token in enumerate(itertools.chain(*ocr_tokens))}
        flattened_ocr_features = torch.cat([feature for feature in ocr_features], dim=0) # (ocr_len, d_model)
        
        # match answers to fixed vocabulary and OCR tokens.
        ocr2inds = defaultdict(list)
        for idx, token in flattened_ocr_tokens.items():
            ocr2inds[token].append(idx)

        # get all possible sequences of indices of texts
        text_inds = [self.match_text_to_indices(text, self.vocab.stoi, ocr2inds) for text in list_of_texts]
        
        # randomly select representation for texts
        selected_text_inds = [matched_ids[np.random.choice(len(matched_ids))] for matched_ids in text_inds]
        tokens = torch.tensor(selected_text_inds).long().to(ocr_features.device)
        padding_mask = generate_padding_mask(tokens, padding_idx=self.vocab.padding_ids).to(ocr_features.device)
        seq_len = tokens.shape[1]
        sequential_mask = generate_sequential_mask(seq_len).to(ocr_features.device)

        # construct the dynamic embeding weights
        weights = torch.cat([self.fixed_weights, flattened_ocr_features], dim=0) # (vocab_len + ocr_len, d_model)

        features = F.embedding(tokens, weights, padding_idx=self.vocab.padding_ids)

        return features, (padding_mask, sequential_mask)

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
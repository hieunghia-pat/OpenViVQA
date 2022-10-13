import torch
from torch import nn
from torch.nn import functional as F

from data_utils.vocab import Vocab
from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel, T5Tokenizer, T5EncoderModel

import os
import numpy as np
from typing import List, Union
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
class OcrEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.padding_idx = 0
        self.device = config.DEVICE

        if config.WORD_EMBEDDING is None: # define the customized vocab
            ocr_texts = []
            for file in os.listdir(config.OCR_PATH):
                ocr_features = np.load(os.path.join(config.OCR_PATH, file), allow_pickle=True)[()]
                ocr_texts.extend(ocr_features["texts"])
            ocr_texts = set(ocr_texts)
            self.stoi = {token: i for i, token in enumerate(ocr_texts)}
            self.stoi.update({vocab.ocr_token: len(self.stoi)})
            self.itos = {i: token for token, i in self.stoi.items()}
            self.word_embedding = None
            self.components = nn.Embedding(len(ocr_texts), config.D_MODEL, self.padding_idx)
        else:
            self.word_embedding = build_word_embedding(config)
            self.stoi = self.word_embedding.stoi
            self.stoi.update({vocab.ocr_token: len(self.stoi)})
            self.itos = self.word_embedding.itos
            self.itos.update({len(self.stoi): vocab.ocr_token})
            self.components = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings=self.word_embedding.vectors, freeze=True, padding_idx=self.padding_idx),
                nn.Linear(config.D_EMBEDDING, config.D_MODEL),
                nn.Dropout(config.DROPOUT)
            )

    def forward(self, texts: List[List[str]]):
        bs = len(texts)
        max_len = max([len(text) for text in texts])
        tokens = torch.zeros((bs, max_len)).long().to(self.device)
        for batch, text in enumerate(texts):
            for idx, token in enumerate(text):
                if token in self.stoi:
                    tokens[batch][idx] = self.stoi[token]
                else:
                    tokens[batch][idx] = self.stoi["ocr"]
        padding_masks = generate_padding_mask(tokens, padding_idx=self.padding_idx).to(tokens.device)
        seq_len = tokens.shape[-1]
        sequential_masks = generate_sequential_mask(seq_len).to(tokens.device)

        features = self.components(tokens)

        return features, (padding_masks, sequential_masks)

@META_TEXT_EMBEDDING.register()
class DynamicEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.d_model = config.D_MODEL
        self.vocab = vocab

        self.register_parameter("fixed_weights", nn.parameter.Parameter(nn.init.xavier_uniform_(torch.ones((len(vocab), self.d_model)))))

    def match_text_to_indices(self, text: List[str], vocab2idx, oov2inds):
        """
            Match an text to a list of sequences of indices
            each index corresponds to either a fixed vocabulary or an OOV token
            (in the index address space, the OOV tokens are after the fixed vocab)
        """
        answer_word_matches = []
        for word in text:
            # match word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx:
                matched_inds.append(vocab2idx[word])
            # match answer word to OOV
            if word in oov2inds:
                matched_inds.extend(oov2inds[word])
            if matched_inds == []:
                print(word)
                raise
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [
                seq + (idx,)
                for seq in idx_seq_list for idx in matched_inds
            ]

        return idx_seq_list

    def pad_oov_tokens(self, oov_tokens: List[List[str]], padding_token):
        padded_oov_tokens = []
        max_len = max([len(oov) for oov in oov_tokens])
        for oov in oov_tokens:
            if max_len > len(oov):
                oov.extend([padding_token]*(max_len - len(oov)))
            padded_oov_tokens.append(oov)

        return padded_oov_tokens

    def encode_sequence(self, list_of_text: List[List[str]], padding_token):
        padded_list_of_text = []
        max_len = max([len(text)+2 for text in list_of_text])
        for text in list_of_text:
            text = [self.vocab.bos_token] + text + [self.vocab.eos_token]
            if max_len - len(text) > 0:
                text.extend([padding_token] * (max_len - len(text)))
            padded_list_of_text.append(text)

        return padded_list_of_text

    def forward(self, list_of_texts: Union[List[List[str]], torch.Tensor], oov_tokens: List[List[str]], oov_features: torch.Tensor):
        if isinstance(list_of_texts, list): # is we have not encode the texts
            list_of_texts = self.encode_sequence(list_of_texts, self.vocab.padding_token)
            oov_tokens = self.pad_oov_tokens(oov_tokens, padding_token=self.vocab.ocr_token)
            flattened_oov_tokens = {len(self.vocab) + idx: token for idx, token in enumerate(itertools.chain(*oov_tokens))}
            flattened_oov_features = torch.cat([feature for feature in oov_features], dim=0) # (ocr_len, d_model)
            
            # match answers to fixed vocabulary and OCR tokens
            oov2inds = defaultdict(list)
            for idx, token in flattened_oov_tokens.items():
                oov2inds[token].append(idx)

            # get all possible sequences of indices of texts
            text_inds = [self.match_text_to_indices(text, self.vocab.stoi, oov2inds) for text in list_of_texts]
            
            # randomly select representation for texts
            selected_text_inds = [matched_ids[np.random.choice(len(matched_ids))] for matched_ids in text_inds]

            tokens = torch.tensor(selected_text_inds).long().to(oov_features.device)
            shifted_right_tokens = tokens[:, 1:]
            tokens = tokens[:, :-1]
            padding_mask = generate_padding_mask(tokens, padding_idx=self.vocab.padding_idx).to(oov_features.device)
            seq_len = tokens.shape[1]
            sequential_mask = generate_sequential_mask(seq_len).to(oov_features.device)

            # construct the dynamic embeding weights
            weights = torch.cat([self.fixed_weights, flattened_oov_features], dim=0) # (vocab_len + ocr_len, d_model)

            features = F.embedding(tokens, weights, padding_idx=self.vocab.padding_idx)

            return shifted_right_tokens, features, (padding_mask, sequential_mask)
        else:
            assert isinstance(list_of_texts, torch.Tensor), "passed list_of_text must be list or tensor"
            flattened_oov_tokens = {len(self.vocab) + idx: token for idx, token in enumerate(itertools.chain(*oov_tokens))}
            flattened_oov_features = torch.cat([feature for feature in oov_features], dim=0) # (ocr_len, d_model)
            
            tokens = list_of_texts
            padding_mask = generate_padding_mask(tokens, padding_idx=self.vocab.padding_idx).to(oov_features.device)
            seq_len = tokens.shape[1]
            sequential_mask = generate_sequential_mask(seq_len).to(oov_features.device)

            weights = torch.cat([self.fixed_weights, flattened_oov_features], dim=0) # (vocab_len + ocr_len, d_model)

            features = F.embedding(tokens, weights, padding_idx=self.vocab.padding_idx)

            return None, features, (padding_mask, sequential_mask)

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
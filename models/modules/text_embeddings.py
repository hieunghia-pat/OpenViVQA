import torch
from torch import nn
from torch.nn import functional as F

from builders.text_embedding_builder import META_TEXT_EMBEDDING
from builders.word_embedding_builder import build_word_embedding
from models.utils import generate_sequential_mask, generate_padding_mask

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)

from transformers.models.albert.modeling_albert import (
    AlbertConfig,
    AlbertEmbeddings,
    AlbertTransformer,
    AlbertPreTrainedModel
)

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaPreTrainedModel
)

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Config,
    DebertaV2Embeddings,
    DebertaV2Encoder,
    DebertaV2PreTrainedModel
)

from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaConfig,
    XLMRobertaEmbeddings,
    XLMRobertaEncoder,
    XLMRobertaPreTrainedModel
)

from transformers import (
    BertTokenizer,
    AlbertTokenizer, 
    RobertaTokenizer,
    DebertaTokenizer,
    XLMTokenizer
)

from typing import Dict, List
import itertools
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
class OcrWordEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE
        self.padding_idx = 0
        self.padding_token = vocab.padding_token
        self.d_model = config.D_MODEL
        self.d_embedding = config.D_EMBEDDING
        self.word_embedding = build_word_embedding(config)

        self.fc = nn.Linear(config.D_EMBEDDING, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)
    
    def load_word_embedding(self, stoi: Dict[str, int]):
        weights = torch.Tensor(len(stoi), self.word_embedding.dim).to(self.device)
        for token, idx in stoi.items():
            weights[idx] = self.word_embedding[token.strip()]

        return weights

    def forward(self, batch_of_texts: List[List[str]]):
        max_len = max([len(text) for text in batch_of_texts])
        for batch, texts in enumerate(batch_of_texts):
            if len(texts) < max_len:
                texts.extend([self.padding_token] * (max_len-len(texts)))
            batch_of_texts[batch] = texts

        ocr_tokens = []
        for texts in batch_of_texts:
            ocr_tokens.extend(itertools.chain(*[text.strip().split() for text in texts]))
        ocr_tokens = set(ocr_tokens)
        ocr2idx = {token: idx for idx, token in enumerate(ocr_tokens)}

        weights = self.load_word_embedding(ocr2idx)
        weights.requires_grad = False # freeze the embedding weights

        features = deepcopy(batch_of_texts)
        for batch, texts in enumerate(batch_of_texts):
            for idx, token in enumerate(texts):
                token = [ocr2idx[subtoken] for subtoken in token.split()]
                token = torch.tensor(token).long().unsqueeze(0).to(self.device)
                feature = F.embedding(token, weights, padding_idx=self.padding_idx).sum(dim=1)
                features[batch][idx] = feature
            features[batch] = torch.cat(features[batch], dim=0).unsqueeze(0)
        features = torch.cat(features, dim=0)

        features = self.fc(features)
        features = self.dropout(features)

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
class FixedVocabDynamicEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.d_model = config.D_MODEL
        self.vocab = vocab

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

    def forward(self, tokens: torch.Tensor, oov_features: torch.Tensor, fixed_weights):
        padding_mask = generate_padding_mask(tokens, padding_idx=self.vocab.padding_idx).to(oov_features.device)
        seq_len = tokens.shape[1]
        sequential_mask = generate_sequential_mask(seq_len).to(oov_features.device)

        # construct the dynamic embeding weights
        bs = tokens.shape[0]
        fixed_weights = fixed_weights.unsqueeze(0).expand((bs, -1, -1)) # (bs, vocab_len, d_model)
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
class HierarchicalFeaturesExtractor(nn.Module):
    def __init__(self, config, vocab) -> None:
        super().__init__()

        self.embedding = UsualEmbedding(config, vocab)

        self.ngrams = config.N_GRAMS
        self.convs = nn.ModuleList()
        for ngram in self.ngrams:
            self.convs.append(
                nn.Conv1d(in_channels=config.D_MODEL, out_channels=config.D_MODEL, kernel_size=ngram)
            )

        self.reduce_features = nn.Linear(config.D_MODEL, config.D_MODEL)

    def forward(self, tokens: torch.Tensor):
        features, (padding_masks, sequential_masks) = self.embedding(tokens)

        ngrams_features = []
        for conv in self.convs:
            ngrams_features.append(conv(features.permute((0, -1, 1))).permute((0, -1, 1)))
        
        features_len = features.shape[-1]
        unigram_features = ngrams_features[0]
        # for each token in the unigram
        for ith in range(features_len):
            # for each n-gram, we ignore the unigram
            for ngram in range(1, max(self.ngrams)):
                # summing all possible n-gram tokens into the unigram
                for prev_ith in range(max(0, ith-ngram+1), min(ith+1, ngrams_features[ngram].shape[1])):
                    unigram_features[:, ith] += ngrams_features[ngram][:, prev_ith]

        return unigram_features, (padding_masks, sequential_masks)

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class BertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        bert_config = BertConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextBert(bert_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextAlbert(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = AlbertEmbeddings(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = AlbertTransformer(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        embedded_inputs = self.embeddings(txt_inds)
        encoder_inputs = self.embedding_hidden_mapping_in(embedded_inputs)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class AlbertEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        albert_config = AlbertConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = AlbertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextAlbert(albert_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask

class TextRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class RobertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        roberta_config = RobertaConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextRoberta(roberta_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextDeberta_v2(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class DebertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        deberta_config = DebertaV2Config(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = DebertaTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextDeberta_v2(deberta_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
    
class TextXLM(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        
        attention_mask = txt_mask
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            encoder_inputs, attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
    
@META_TEXT_EMBEDDING.register()
class XLMRobertaEmbedding(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.device = config.DEVICE

        xlm_config = XLMRobertaConfig(
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        )

        self.tokenizer = XLMTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = TextXLM(xlm_config)
        if config.LOAD_PRETRAINED:
            self.embedding = self.embedding.from_pretrained(config.PRETRAINED_NAME)
        if config.FREEZE_WEIGHTS:
            # freeze all parameters of pretrained model
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config.HIDDEN_SIZE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
        padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(inputs, padding_mask)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask
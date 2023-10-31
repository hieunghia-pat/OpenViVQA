import torch
from torch import nn
from torch.nn import functional as F

from utils.instance import InstanceList
from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.encoder_builder import build_encoder

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(config.D_MODEL, 1)

    def forward(self, features: torch.Tensor):
        output = self.dropout(self.relu(self.fc1(features)))
        output = self.fc2(output)

        return output

class HierarchicalFeaturesExtractor(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.ngrams = config.N_GRAMS
        self.convs = nn.ModuleList()
        for ngram in self.ngrams:
            self.convs.append(
                nn.Conv1d(in_channels=config.WORD_EMBEDDING_DIM, out_channels=config.D_MODEL, kernel_size=ngram)
            )

        self.reduce_features = nn.Linear(config.D_MODEL, config.D_MODEL)

    def forward(self, features: torch.Tensor):
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

        return unigram_features

@META_ARCHITECTURE.register()
class HierarchicalCoAttention(nn.Module):
    def __init__(self, config, vocab) -> None:
        super().__init__()

        # embedding module
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.question_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        
        # hierarchical feature extractors for texts
        self.hierarchical_extractor = HierarchicalFeaturesExtractor(config.HIERARCHICAL)

        # co-attention module
        self.encoder = build_encoder(config.ENCODER)

        # attributes reduction and classifier
        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.classify = nn.Linear(config.D_MODEL, vocab.total_answers)

    def forward(self, input_features: InstanceList):
        # embedding input features
        vision_features, vision_padding_masks = self.vision_embedding(input_features.region_features)
        text_features, (text_padding_masks, _) = self.question_embedding(input_features.question_tokens)

        # performing hierarchical feature extraction
        text_features = self.hierarchical_extractor(text_features)

        # performing co-attention
        vision_features, text_features = self.encoder(vision_features, vision_padding_masks, 
                                                        text_features, text_padding_masks)
        
        attended_vision_features = self.vision_attr_reduce(vision_features)
        attended_vision_features = F.softmax(attended_vision_features, dim=1)
        attended_text_features = self.text_attr_reduce(text_features)
        attended_text_features = F.softmax(attended_text_features, dim=1)

        weighted_vision_features = (vision_features * attended_vision_features).sum(dim=1)
        weighted_text_features = (text_features * attended_text_features).sum(dim=1)

        output = self.layer_norm(self.vision_proj(weighted_vision_features) + self.text_proj(weighted_text_features))
        output = self.classify(output)

        return F.log_softmax(output, dim=-1)
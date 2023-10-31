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

@META_ARCHITECTURE.register()
class ParallelAttentionTransformer(nn.Module):
    def __init__(self, config, vocab) -> None:
        super().__init__()

        # embedding module
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.question_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

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
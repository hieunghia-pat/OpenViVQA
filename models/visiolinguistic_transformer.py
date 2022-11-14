import torch
from torch import nn
from torch.nn import functional as F

from .base_transformer import BaseTransformer
from utils.instance import InstanceList
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

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
class VisiolinguisticTransformer(BaseTransformer):
    '''
        This model is designed follow the idea of ViLBERT (https://arxiv.org/pdf/1908.02265.pdf).
    '''
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.encoder = build_encoder(config.ENCODER)
        
        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.classify = nn.Linear(config.D_MODEL, vocab.total_answers)

    def forward(self, input_features: InstanceList):
        region_features = input_features.region_features
        region_features, region_padding_mask = self.region_embedding(region_features)
        region_feat_tokens = torch.ones((region_features.shape[0], region_features.shape[1])).long().to(region_features.device) * self.vocab.feat_idx
        region_feat_embedded, _ = self.text_embedding(region_feat_tokens)
        region_features += region_feat_embedded

        region_boxes = input_features.region_boxes
        region_boxes, region_boxes_padding_mask = self.box_embedding(region_boxes)
        region_box_tokens = torch.ones((region_boxes.shape[0], region_boxes.shape[1])).long().to(region_boxes.device) * self.vocab.box_idx
        region_box_embedded, _ = self.text_embedding(region_box_tokens)
        region_boxes += region_box_embedded

        grid_features = input_features.grid_features
        grid_features, grid_padding_mask = self.grid_embedding(grid_features)
        grid_feat_tokens = torch.ones((grid_features.shape[0], grid_features.shape[1])).long().to(grid_features.device) * self.vocab.feat_idx
        grid_feat_embedded, _ = self.text_embedding(grid_feat_tokens)
        grid_features += grid_feat_embedded
        
        grid_boxes = input_features.grid_boxes
        grid_boxes, grid_boxes_padding_mask = self.box_embedding(grid_boxes)
        grid_box_tokens = torch.ones((grid_boxes.shape[0], grid_boxes.shape[1])).long().to(grid_boxes.device) * self.vocab.box_idx
        grid_box_embedded, _ = self.text_embedding(grid_box_tokens)
        grid_boxes += grid_box_embedded
        
        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_padding_mask = torch.cat([region_padding_mask, region_boxes_padding_mask, grid_padding_mask, grid_boxes_padding_mask], dim=-1)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        # Cross-Modality attention
        vision_features, text_features = self.encoder(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            boxes=input_features.boxes,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        )

        # Multimodal fusion
        attended_vision_features = self.vision_attr_reduce(vision_features)
        attended_vision_features = F.softmax(attended_vision_features, dim=1)
        attended_text_features = self.text_attr_reduce(text_features)
        attended_text_features = F.softmax(attended_text_features, dim=1)

        weighted_vision_features = (vision_features * attended_vision_features).sum(dim=1)
        weighted_text_features = (text_features * attended_text_features).sum(dim=1)

        output = self.layer_norm(self.vision_proj(weighted_vision_features) + self.text_proj(weighted_text_features))
        output = self.classify(output)

        return output

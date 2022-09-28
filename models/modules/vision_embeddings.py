
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from typing import List

from builders.vision_embedding_builder import META_VISION_EMBEDDING
from models.utils import generate_padding_mask

@META_VISION_EMBEDDING.register()
class FeatureEmbedding(nn.Module):
    def __init__(self, config):
        super(FeatureEmbedding, self).__init__()

        self.proj = nn.Linear(config.D_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, features):
        masks = generate_padding_mask(features, padding_idx=0).to(features.device)

        features = self.gelu(self.proj(features))
        features = self.dropout(features)

        return features, masks

@META_VISION_EMBEDDING.register()
class ViTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(config.PRETRAINED_NAME)
        self.backbone = ViTModel.from_pretrained(config.PRETRAINED_NAME)

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, images: List[Image.Image]):
        inputs = self.feature_extractor(images, return_tensors="pt")
        features = self.backbone(**inputs)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        
        return out
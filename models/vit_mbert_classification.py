import torch
from torch import nn
from torch.nn import functional as F

from models.base_classification import BaseClassificationModel
from builders.model_builder import META_ARCHITECTURE
from builders.vision_embedding_builder import build_vision_embedding
from builders.text_embedding_builder import build_text_embedding
from utils.instance import Instance

@META_ARCHITECTURE.register()
class ViTmBERTClassification(BaseClassificationModel):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.vision_encoder = build_vision_embedding(config.VISION_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.norm = nn.LayerNorm(config.D_MODEL)

        self.proj = nn.Linear(config.D_MODEL, vocab.total_answers)

    def forward(self, inputs: Instance):
        images = inputs.image
        questions = inputs.question

        vision_features, _ = self.vision_encoder(images)
        text_features, _ = self.text_embedding(questions)

        fused_features = torch.cat([vision_features, text_features], dim=1)
        fused_features = self.dropout(self.fusion(fused_features))
        out = fused_features.sum(dim=1)
        out = self.proj(out)

        return F.log_softmax(out, dim=-1)
import torch
from torch import nn
from torch.nn import functional as F

from models.base_transformer import BaseTransformer
from builders.model_builder import META_ARCHITECTURE
from builders.vision_embedding_builder import build_vision_embedding
from builders.text_embedding_builder import build_text_embedding
from builders.decoder_builder import build_decoder
from utils.instance import Instance

@META_ARCHITECTURE.register()
class ViTmBERTGeneration(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.vision_encoder = build_vision_embedding(config.VISION_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.norm = nn.LayerNorm(config.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab)

    def forward(self, inputs: Instance):
        fused_features, fused_padding_mask = self.encoder_forward(inputs)
        
        answer_tokens = inputs.answer_tokens
        out = self.decoder(
            answer_tokens=answer_tokens,
            encoder_features=fused_features,
            encoder_attention_mask=fused_padding_mask
        )

        return F.log_softmax(out, dim=-1)

    def encoder_forward(self, inputs: Instance):
        features = inputs.grid_features
        questions = inputs.question

        vision_features, vision_padding_mask = self.vision_encoder(features)
        text_features, text_padding_mask = self.text_embedding(questions)

        fused_features = torch.cat([vision_features, text_features], dim=1)
        fused_features = self.gelu(self.fusion(fused_features))
        fused_features = self.dropout(fused_features)
        fused_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)

        return fused_features, fused_padding_mask
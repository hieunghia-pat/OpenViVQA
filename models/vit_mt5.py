import torch
from torch import nn

from .base_transformer import BaseTransformer
from utils.instance import Instance
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class ViTmT5(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.fusion = nn.Linear(config.MULTIMODAL_FUSION)
        self.norm = nn.LayerNorm(config.MULTIMODAL_FUSION.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab)

    def forward(self, input_features: Instance):
        images = input_features.image
        questions = input_features.question

        vision_features, vision_padding_mask = self.vision_embedding(images) # (3, 49, 512) (3, 1, 1, 49)
        text_features, text_padding_mask = self.text_embedding(questions) # (3, 12, 512) (3, 1, 1, 12)

        encoder_features = torch.cat([vision_features, text_features], dim=1) # (3, 49 + 12, 512)
        encoder_features = self.fusion(encoder_features)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1) # (3, 1, 1, 49 + 12)

        answer_tokens = input_features.answer_tokens
        output = self.decoder(Instance(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=encoder_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instance):
        images = input_features.image
        questions = input_features.question

        vision_features, vision_padding_mask = self.vision_embedding(images) # (3, 49, 512) (3, 1, 1, 49)
        text_features, text_padding_mask = self.text_embedding(questions) # (3, 12, 512) (3, 1, 1, 12)

        encoder_features = torch.cat([vision_features, text_features], dim=1) # (3, 49 + 12, 512)
        encoder_features = self.fusion(encoder_features)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1) # (3, 1, 1, 49 + 12)

        return encoder_features, encoder_padding_mask
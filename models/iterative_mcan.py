import torch
from torch import nn

from .base_transformer import BaseTransformer
from utils.instance import Instance
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class IterativeMCAN(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        self.fusion = PositionWiseFeedForward(config.MULTIMODAL_FUSION)
        self.norm = nn.LayerNorm(config.MULTIMODAL_FUSION.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab=vocab)

    def forward(self, input_features: Instance):
        encoder_features, encoder_padding_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        output = self.decoder(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=encoder_padding_mask
        )

        return output

    def encoder_forward(self, input_features: Instance):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        # SA
        text_features = self.self_encoder(
            features=text_features,
            padding_mask=text_padding_mask
        )

        # GSA
        vision_features = self.guided_encoder(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        )

        # Multimodal fusion
        encoder_features = torch.cat([vision_features, text_features], dim=1)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)
        encoder_features = self.fusion(encoder_features)
        encoder_features = self.norm(encoder_features)

        return encoder_features, encoder_padding_mask

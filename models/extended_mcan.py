import torch
from torch import nn

from .base_transformer import BaseTransformer
from utils.instances import Instances
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class ExtendedMCAN(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)
        self.vocab = vocab

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        self.fusion = PositionWiseFeedForward(config.MULTIMODAL_FUSION)
        self.norm = nn.LayerNorm(config.MULTIMODAL_FUSION.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab=vocab)

    def forward(self, input_features: Instances):
        encoder_features, encoder_padding_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        output = self.decoder(Instances(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=encoder_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instances):
        region_features = input_features.region_features
        region_feat_tokens = torch.ones((region_features.shape[0], region_features.shape[1])).long() * self.vocab.feat_idx
        region_features += self.decoder.word_emb(region_feat_tokens)

        region_boxes = input_features.region_boxes
        region_box_tokens = torch.ones((region_boxes.shape[0], region_boxes.shape[1])).long() * self.vocab.box_idx
        region_boxes += self.decoder.word_emb(region_box_tokens)

        grid_features = input_features.grid_features
        grid_feat_tokens = torch.ones((grid_features.shape[0], region_features.shape[1])).long() * self.vocab.feat_idx
        grid_features += self.decoder.word_emb(grid_feat_tokens)
        
        grid_boxes = input_features.grid_boxes
        grid_box_tokens = torch.ones((grid_boxes.shape[0], region_boxes.shape[1])).long() * self.vocab.box_idx
        grid_boxes += self.decoder.word_emb(grid_box_tokens)

        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        # SA
        text_features = self.self_encoder(Instances(
            features=text_features,
            features_padding_mask=text_padding_mask
        ))

        # GSA
        vision_features = self.guided_encoder(Instances(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        ))

        # Multimodal fusion
        encoder_features = torch.cat([vision_features, text_features], dim=1)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)
        encoder_features = self.fusion(encoder_features)
        encoder_features = self.norm(encoder_features)

        return encoder_features, encoder_padding_mask

@META_ARCHITECTURE.register()
class ExtendedMCANUsingRegion(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        self.fusion = PositionWiseFeedForward(config.MULTIMODAL_FUSION)
        self.norm = nn.LayerNorm(config.MULTIMODAL_FUSION.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab=vocab)

    def forward(self, input_features: Instances):
        encoder_features, encoder_padding_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        output = self.decoder(Instances(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=encoder_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instances):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        # SA
        text_features = self.self_encoder(Instances(
            features=text_features,
            features_padding_mask=text_padding_mask
        ))

        # GSA
        vision_features = self.guided_encoder(Instances(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        ))

        # Multimodal fusion
        encoder_features = torch.cat([vision_features, text_features], dim=1)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)
        encoder_features = self.fusion(encoder_features)
        encoder_features = self.norm(encoder_features)

        return encoder_features, encoder_padding_mask

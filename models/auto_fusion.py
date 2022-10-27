import torch
from torch import nn

from .base_transformer import BaseTransformer
from utils.instances import Instances
from models.modules.encoders import AutoFusionEncoder
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class AutoFusion(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)
        self.vocab = vocab

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.self_encoder = build_encoder(config.SELF_ENCODER)

        self.fusion = AutoFusionEncoder(config.AUTO_FUSION)
        self.norm = nn.LayerNorm(config.AUTO_FUSION.D_MODEL)

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
        region_features, region_padding_mask = self.region_embedding(region_features)
        region_feat_tokens = torch.ones((region_features.shape[0], region_features.shape[1])).long().to(region_features.device) * self.vocab.feat_idx
        region_feat_embedded, _ = self.decoder.word_emb(region_feat_tokens)
        region_features += region_feat_embedded

        region_boxes = input_features.region_boxes
        region_boxes, region_boxes_padding_mask = self.box_embedding(region_boxes)
        region_box_tokens = torch.ones((region_boxes.shape[0], region_boxes.shape[1])).long().to(region_boxes.device) * self.vocab.box_idx
        region_box_embedded, _ = self.decoder.word_emb(region_box_tokens)
        region_boxes += region_box_embedded

        grid_features = input_features.grid_features
        grid_features, grid_padding_mask = self.grid_embedding(grid_features)
        grid_feat_tokens = torch.ones((grid_features.shape[0], grid_features.shape[1])).long().to(grid_features.device) * self.vocab.feat_idx
        grid_feat_embedded, _ = self.decoder.word_emb(grid_feat_tokens)
        grid_features += grid_feat_embedded
        
        grid_boxes = input_features.grid_boxes.squeeze(1)
        grid_boxes, grid_boxes_padding_mask = self.box_embedding(grid_boxes)
        grid_box_tokens = torch.ones((grid_boxes.shape[0], grid_boxes.shape[1])).long().to(grid_boxes.device) * self.vocab.box_idx
        grid_box_embedded, _ = self.decoder.word_emb(grid_box_tokens)
        grid_boxes += grid_box_embedded.squeeze(1)

        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_padding_mask = torch.cat([region_padding_mask, region_boxes_padding_mask, grid_padding_mask, grid_boxes_padding_mask], dim=-1)

        question = input_features.question
        text_features, text_padding_mask = self.text_embedding(question)
        
        # SA
        text_features = self.self_encoder(Instances(
            features=text_features,
            features_padding_mask=text_padding_mask
        ))
        # SA
        vision_features = self.self_encoder(Instances(
            features=vision_features,
            features_padding_mask=vision_padding_mask
        ))

        # Multimodal fusion
        encoder_features = torch.cat([vision_features, text_features], dim=1)
        encoder_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)
        encoder_features = self.fusion(encoder_features)
        encoder_features = self.norm(encoder_features)

        return encoder_features, encoder_padding_mask
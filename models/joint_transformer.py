import torch

from .base_transformer import BaseTransformer
from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class JointTransformer(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.encoder = build_encoder(config.ENCODER)

        self.decoder = build_decoder(config.DECODER, vocab=vocab)

    def forward(self, input_features: Instance):
        encoder_features, encoder_padding_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        output = self.decoder(Instance(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=encoder_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instance):
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
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long().to(question_tokens.device) * self.vocab.question_idx
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)
        q_embedded, _ = self.text_embedding(q_tokens)
        text_features += q_embedded

        joint_features = torch.cat([vision_features, text_features], dim=1)
        joint_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)

        # Joint-Modality attention
        encoder_features = self.encoder(Instance(
            features=joint_features,
            features_padding_mask=joint_padding_mask
        ))

        return encoder_features, joint_padding_mask

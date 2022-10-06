import torch
from torch import nn
from torch.nn import functional as F

from .base_unique_transformer import BaseUniqueTransformer
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class M4C(BaseUniqueTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)
        self.d_model = config.D_MODEL

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.ocr_det_embedding = build_vision_embedding(config.OCR_DET_EMBEDDING)
        self.ocr_rec_embedding = build_vision_embedding(config.OCR_REC_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.self_encoder = build_encoder(config.SELF_ENCODER)        

    def embed_features(self, input_features: Instances):
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

        ocr_det_features = input_features.ocr_det_features
        ocr_det_features, ocr_det_padding_mask = self.ocr_det_embedding(ocr_det_features)
        ocr_det_tokens = torch.ones((ocr_det_features.shape[0], ocr_det_features.shape[1])).long().to(ocr_det_features.device) * self.vocab.ocr_det_idx
        ocr_det_embedded, _ = self.text_embedding(ocr_det_tokens)
        ocr_det_features += ocr_det_embedded

        ocr_rec_features = input_features.ocr_rec_features
        ocr_rec_features, ocr_rec_padding_mask = self.ocr_rec_embedding(ocr_rec_features)
        ocr_rec_tokens = torch.ones((ocr_rec_features.shape[0], ocr_rec_features.shape[1])).long().to(ocr_rec_features.device) * self.vocab.ocr_rec_idx
        ocr_rec_embedded, _ = self.text_embedding(ocr_rec_tokens)
        ocr_rec_features += ocr_rec_embedded

        ocr_boxes = input_features.ocr_boxes
        ocr_boxes, ocr_boxes_padding_mask = self.box_embedding(ocr_boxes)
        ocr_box_tokens = torch.ones((ocr_boxes.shape[0], ocr_boxes.shape[1])).long().to(ocr_boxes.device) * self.vocab.box_idx
        ocr_box_embedded, _ = self.text_embedding(ocr_box_tokens)
        ocr_det_features += ocr_box_embedded

        ocr_tokens = input_features.ocr_tokens
        ocr_features, ocr_padding_mask = self.text_embedding(ocr_tokens)
        ocr_embedding_tokens = torch.ones((ocr_tokens.shape[0], ocr_tokens.shape[1])).long().to(ocr_tokens.device) * self.vocab.ocr_idx
        ocr_embedded, _ = self.text_embedding(ocr_embedding_tokens)
        ocr_features += ocr_embedded

        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes, ocr_det_features, ocr_rec_features, ocr_boxes, ocr_features], dim=1)
        vision_padding_mask = torch.cat([region_padding_mask, region_boxes_padding_mask, grid_padding_mask, grid_boxes_padding_mask, ocr_det_padding_mask, ocr_rec_padding_mask, ocr_boxes_padding_mask, ocr_padding_mask], dim=-1)

        question_tokens = input_features.question_tokens
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long().to(question_tokens.device) * self.vocab.question_idx
        question_features, (question_padding_mask, _) = self.text_embedding(question_tokens)
        q_embeded, _ = self.text_embedding(q_tokens)
        question_features += q_embeded

        joint_features = torch.cat([vision_features, question_features], dim=1)
        joint_padding_mask = torch.cat([vision_padding_mask, question_padding_mask], dim=-1)

        joint_features_len = joint_features.shape[1]
        joint_attention_mask = joint_padding_mask.expand((-1, -1, joint_features_len, -1)) # (bs, 1, joint_features_len, joint_features_len)

        return joint_features, (joint_padding_mask, joint_attention_mask)

    def forward(self, input_features: Instances):
        joint_features, (joint_padding_mask, joint_attention_mask) = self.embed_features(input_features)
        joint_features_len = joint_features.shape[1]
        answer_tokens = input_features.answer_tokens
        joint_features, (joint_padding_mask, joint_attention_mask) = self.append_answer(joint_features, (joint_padding_mask, joint_attention_mask), answer_tokens)

        out = self.encoder(Instances(
            features=joint_features,
            features_padding_mask=joint_padding_mask,
            features_attention_mask=joint_attention_mask
        ))
        out = out[:, joint_features_len:]
        out = self.fc(out)

        return F.log_softmax(out, dim=-1)

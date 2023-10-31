import torch
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
class VisionOcrEmbedding(nn.Module):
    def __init__(self, config):
        super(VisionOcrEmbedding, self).__init__()

        self.linear_obj_feat_to_mmt_in = nn.Linear(config.D_OBJ_FEATURE, config.D_MODEL)
        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)
        self.obj_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_gelu = nn.GELU()
        self.obj_dropout = nn.Dropout(config.DROPOUT)

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            config.D_OCR_FEATURE, config.D_MODEL
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        # OCR word embedding features
        # self.ocr_word_embedding = build_word_embedding(self.config.OCR_TEXT_EMBEDDING)

        self.ocr_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_text_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_gelu = nn.GELU()
        self.ocr_drop = nn.Dropout(config.DROPOUT)

    def forward(self, obj_features: torch.Tensor, obj_boxes: torch.Tensor, ocr_det_features: torch.Tensor, 
                ocr_rec_features: torch.Tensor, ocr_fasttext: torch.Tensor, ocr_boxes: torch.Tensor):
        ocr_features = torch.cat([ocr_det_features, ocr_rec_features, ocr_fasttext], dim=-1)

        obj_masks = generate_padding_mask(obj_features, padding_idx=0).to(obj_features.device)
        ocr_masks = generate_padding_mask(ocr_det_features, padding_idx=0).to(ocr_features.device)
        masks = torch.cat([obj_masks, ocr_masks], dim=-1)

        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_features)
        ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_boxes)
        )
        obj_mmt_in = self.obj_dropout(
            self.obj_gelu(obj_mmt_in)
        )

        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_features)
        ) + self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_boxes)
        )
        ocr_mmt_in = self.ocr_drop(
            self.ocr_gelu(ocr_mmt_in)
        )

        features = torch.cat([obj_mmt_in, ocr_mmt_in], dim=1)

        return features, masks

@META_VISION_EMBEDDING.register()
class ViTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.DEVICE)

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(config.PRETRAINED_NAME)
        self.backbone = ViTModel.from_pretrained(config.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, images: List[Image.Image]):
        inputs = self.feature_extractor(images, return_tensors="pt").to(self.device)
        features = self.backbone(**inputs).last_hidden_state
        padding_mask = generate_padding_mask(features, padding_idx=0)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        
        return out, padding_mask
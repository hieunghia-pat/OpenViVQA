import torch
from torch import nn
import numpy as np

from builders.vision_embedding_builder import META_VISION_EMBEDDING
from models.utils import generate_padding_mask
from models.modules.attentions import ScaledDotProductAttention

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
class SemanticOCREmbedding(nn.Module):
    def __init__(self, config, embedding: nn.Module) -> None:
        super().__init__()

        self.embedding = embedding

        self.linear_det_features = nn.Linear(config.ocr_embedding.d_det, config.d_model)
        self.linear_rec_features = nn.Linear(config.ocr_embedding.d_rec, config.d_model)
        self.linear_boxes = nn.Linear(4, config.d_model)

        self.layer_norm_det = nn.LayerNorm(config.d_model)
        self.layer_norm_rec = nn.LayerNorm(config.d_model)
        self.layer_norm_bboxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.device = torch.device(f'{config.device}' if torch.cuda.is_available() else 'cpu')

    def forward(self,
                det_features: torch.FloatTensor,
                rec_features: torch.FloatTensor,
                bboxes: torch.FloatTensor,
                ocr_token_ids: torch.FloatTensor):
        
        ocr_token_ids = ocr_token_ids.to(torch.long)
        ocr_text_emb = self.embedding(ocr_token_ids)
        embedding_mask = (ocr_token_ids == 0).unsqueeze(-1)
        ocr_text_emb = ocr_text_emb.masked_fill(embedding_mask, 0)
        ocr_text_emb = ocr_text_emb.sum(dim=-2)
    
        ocr_feature_emb = (self.layer_norm_det(self.linear_det_features(det_features))+
                        self.layer_norm_rec(self.linear_rec_features(rec_features)))
        ocr_box_emb = self.layer_norm_bboxes(self.linear_boxes(bboxes))
        
        ocr_features = ocr_feature_emb + ocr_box_emb + ocr_text_emb
        ocr_features = self.dropout(self.gelu(ocr_features))

        mask = generate_padding_mask(det_features, padding_idx=0)
        mask = torch.logical_not(mask)

        return ocr_features, mask
    
@META_VISION_EMBEDDING.register()
class SemanticObjectEmbedding(nn.Module):
    def __init__(self, config, embedding: nn.Module) -> None:
        super().__init__()

        self.device = torch.device(f'{config.device}' if torch.cuda.is_available() else 'cpu')

        self.embedding = embedding

        self.linear_region_features = nn.Linear(config.object_embedding.d_feature, config.d_model)
        self.linear_region_boxes = nn.Linear(4, config.d_model)

        self.layer_norm_region = nn.LayerNorm(config.d_model)
        self.layer_norm_region_boxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                features: torch.FloatTensor,
                bboxes: torch.FloatTensor,
                tag_ids: torch.LongTensor) -> torch.Tensor:

        tag_ids = tag_ids.to(torch.long)
        tag_embs = self.embedding(tag_ids)
        embedding_mask = (tag_ids == 0).unsqueeze(-1)
        tag_embs = tag_embs.masked_fill(embedding_mask, 0)
        tag_embs = tag_embs.sum(dim=-2)

        mask = generate_padding_mask(features, padding_idx=0)
        mask = torch.logical_not(mask)

        features = self.linear_region_features(features)
        bboxes = self.linear_region_boxes(bboxes)
        
        features = self.layer_norm_region(features)
        bboxes = self.layer_norm_region_boxes(bboxes)

        obj_features = features + bboxes + tag_embs
        obj_features = self.dropout(self.gelu(obj_features))

        return obj_features, mask

@META_VISION_EMBEDDING.register()
class SpatialCirclePosition(ScaledDotProductAttention):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.dist_embedding = nn.Embedding(
            num_embeddings=config.ocr_embedding.num_distance,
            embedding_dim=config.num_heads
        )

    def patch(self, ocr_boxes: torch.Tensor, image_sizes: torch.Tensor) -> tuple:
        """
            ocr_boxes: (bs, n_ocr, 4)
            image_sizes: (bs, 4)
            return: (bs, x_centroid, y_centroid)
        """
        
        size_per_area = (image_sizes[:, :2] // 11).unsqueeze(1) # (bs, 1, 2)
        lower_bounds = torch.arange(
            start=0, 
            end=11, 
            step=1
        ).unsqueeze(0).repeat(ocr_boxes.shape[0], 1).to(ocr_boxes.device) # (bs, 11)
        higher_bounds = lower_bounds + 1
        # width boundaries
        width_lower_bounds = lower_bounds * size_per_area[:, :, 0]
        width_higher_bounds = higher_bounds * size_per_area[:, :, 0]
        # height boundaries
        height_lower_bounds = lower_bounds * size_per_area[:, :, 1]
        height_higher_bounds = higher_bounds * size_per_area[:, :, 1]

        # reshape the bounds so that we can broadcast the dimension
        width_lower_bounds = width_lower_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        width_higher_bounds = width_higher_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        height_lower_bounds = height_lower_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        height_higher_bounds = height_higher_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        ocr_boxes = ocr_boxes.unsqueeze(-2) # (bs, n_ocr, 1, 4)
        ocr_x_centroid = (ocr_boxes[:, :, :, 0] + ocr_boxes[:, :, :, 2]) // 2
        ocr_y_centroid = (ocr_boxes[:, :, :, 1] + ocr_boxes[:, :, :, 3]) // 2
        selected_x_centroid = torch.logical_and(torch.le(width_lower_bounds, ocr_x_centroid), torch.le(ocr_x_centroid, width_higher_bounds)) # (bs, n_ocr, 11)
        selected_y_centroid = torch.logical_and(torch.le(height_lower_bounds, ocr_y_centroid), torch.le(ocr_y_centroid, height_higher_bounds)) # (bs, n_ocr, 11)
        # determine the appropriate patch
        selected_x_centroid = selected_x_centroid.long().argmax(dim=-1) # (bs, n_ocr)
        selected_y_centroid = selected_y_centroid.long().argmax(dim=-1) # (bs, n_orc)

        return selected_x_centroid, selected_y_centroid
    
    def pytha(self, p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
        """
            p_i: (bs, *, 2)
            p_j: (bs, *, 2)
            return: (bs, *)
        """
        delta = p_i - p_j
        return torch.round(torch.sqrt(torch.square(delta).sum(dim=-1))).long()

    def forward(self,
                ocr_features: torch.Tensor,
                ocr_boxes: torch.Tensor, 
                ocr_padding_masks: torch.Tensor, 
                image_sizes: torch.Tensor
        ) -> torch.Tensor:
        """
            ocr_features: (bs, n_ocr, d_model)
            ocr_boxes: (bs, n_ocr, 4)
            ocr_padding_masks: (bs, n_ocr)
            image_sizes: (bs, 4)
        """

        bs, nq, _ = ocr_boxes.shape
        ocr_boxes = (ocr_boxes * image_sizes.unsqueeze(1))

        patch_x, patch_y = self.patch(ocr_boxes, image_sizes)
        patch_i_x = patch_x.unsqueeze(-1).repeat(1, 1, nq).unsqueeze(-1) # (bs, n_ocr, n_ocr, 1)
        patch_i_y = patch_y.unsqueeze(-1).repeat(1, 1, nq).unsqueeze(-1) # (bs, n_ocr, n_ocr, 1)
        patch_i = torch.cat([patch_i_x, patch_i_y], dim=-1) # (bs, n_ocr, n_ocr, 2)
        patch_j_x = patch_x.unsqueeze(-2).repeat(1, nq, 1).unsqueeze(-1) # (bs, n_ocr, n_ocr, 1)
        patch_j_y = patch_y.unsqueeze(-2).repeat(1, nq, 1).unsqueeze(-1) # (bs, n_ocr, n_ocr, 1)
        patch_j = torch.cat([patch_j_x, patch_j_y], dim=-1) # (bs, n_ocr, n_ocr, 2)
        dist = self.pytha(patch_i, patch_j) # (bs, n_ocr, n_ocr)
        dist = self.dist_embedding(dist).view(bs, nq, nq, -1).permute((0, -1, 1, 2)) # (bs, h, nq, nq)

        q = self.fc_q(ocr_features).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (bs, h, nq, d_k)
        k = self.fc_k(ocr_features).view(bs, nq, self.h, self.d_k).permute(0, 2, 3, 1)  # (bs, h, d_k, nk)
        v = self.fc_v(ocr_features).view(bs, nq, self.h, self.d_v).permute(0, 2, 1, 3)  # (bs, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (bs, h, nq, nq)
        att += ocr_padding_masks.unsqueeze(1).unsqueeze(1)
        att = torch.softmax(att + dist, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(bs, nq, self.h * self.d_v)  # (bs, nq, h*d_v)
        out = self.fc_o(out)  # (bs, nq, d_model)

        return out
    
@META_VISION_EMBEDDING.register()
class TextSemanticSeparate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.device = torch.device(config.device)
        self.context_emb = nn.Parameter(torch.zeros(1, config.d_model))
        nn.init.xavier_uniform_(self.context_emb)

    def forward(self,
                ocr_emb: torch.Tensor,
                ocr_box_emb: torch.Tensor,
                ocr_text_emb: torch.Tensor
            ) -> torch.Tensor:
        """
            ocr_emb: (bs, n_ocr, d_model)
            ocr_box_emb: (bs, n_ocr, d_model)
            ocr_text_emb: (bs, n_ocr, d_model)
        """
        bs, n_ocr, d_ocr = ocr_emb.shape
        extended_ocr_emb = torch.zeros((bs, n_ocr*2, d_ocr)).to(self.device)
        extended_ocr_emb[:, 0::2] = ocr_emb
        extended_ocr_emb[:, 1::2] = ocr_emb

        extended_ocr_box_emb = torch.zeros((bs, n_ocr*2, d_ocr)).to(self.device)
        extended_ocr_box_emb[:, 0::2] = ocr_box_emb
        extended_ocr_box_emb[:, 1::2] = ocr_box_emb

        extended_ocr_text_emb = torch.zeros((bs, n_ocr*2, d_ocr)).to(self.device)
        extended_ocr_text_emb[:, 0::2] = ocr_text_emb
        extended_ocr_text_emb[:, 1::2] = self.context_emb

        tss_features = extended_ocr_emb + extended_ocr_box_emb + extended_ocr_text_emb

        return tss_features[:, 0::2] + tss_features[:, 1::2]

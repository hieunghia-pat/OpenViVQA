import torch
from torch import nn

import numpy as np
from typing import List

from models.modules.attentions import ScaledDotProductAttention

class SpatialCirclePosition(ScaledDotProductAttention):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.dist_embedding = nn.Embedding(
            num_embeddings=config.NUM_DISTANCE,
            embedding_dim=config.HEAD
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
            ocr_padding_masks: (bs, 1, 1, n_ocr)
            image_sizes: (bs, 4)
        """
        bs, nq, _ = ocr_boxes.shape
        image_sizes = image_sizes
        ocr_boxes = (ocr_boxes * image_sizes.unsqueeze(1)).long()

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

        return out, att
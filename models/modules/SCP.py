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

        self.init_weights()

    def init_weights(self):
        super().init_weights()
        nn.init.xavier_uniform_(self.dist_embedding.weight)

    def patch(ocr_box: tuple,
              image_size: tuple
            ) -> tuple:
        """
            ocr_box: (x1, y1, x2, y2)
            image_size: (w, h)
            return: (x_centroid, y_centroid)
        """
        ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
        ocr_centroid_x = (ocr_x2 - ocr_x1) // 2
        ocr_centroid_y = (ocr_y2 - ocr_y1) // 2

        img_w, img_h = image_size
        
        w_per_area = img_w // 11
        h_per_area = img_h // 11
        x_unit_centroid = w_per_area // 2
        y_unit_centroid = h_per_area // 2
        
        # list of indices
        iths = [i for i in range(11)]
        # list of x boundaries
        x_lower_bounds = np.array([ith*w_per_area for ith in iths])
        x_centroids = np.array([ith*w_per_area+x_unit_centroid for ith in iths])
        x_higher_bounds = np.array([(ith+1)*w_per_area for ith in iths])
        # list of y boundaries
        y_lower_bounds = np.array([ith*h_per_area for ith in iths])
        y_centroids = np.array([ith*h_per_area+y_unit_centroid for ith in iths])
        y_higher_bounds = np.array([(ith+1)*h_per_area for ith in iths])

        selected_x_centroid = None
        selected_y_centroid = None
        for ith in iths:
            x_lower_bound = x_lower_bounds[ith]
            x_higher_bound = x_higher_bounds[ith]
            if x_lower_bound <= ocr_centroid_x <= x_higher_bound:
                selected_x_centroid = x_centroids[ith]

            y_lower_bound = y_lower_bounds[ith]
            y_higher_bound = y_higher_bounds[ith]
            if y_lower_bound <= ocr_centroid_y <= y_higher_bound:
                selected_y_centroid = y_centroids[ith]

        return (selected_x_centroid, selected_y_centroid)
    
    def pytha(self, p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
        """
            p_i: (bs, 2)
            p_j: (bs, 2)
            return: (bs, )
        """
        delta = p_i - p_j
        return torch.sqrt(torch.square(delta[0]) + torch.square(delta[1])).long()

    def forward(self,
                ocr_features: torch.Tensor,
                ocr_boxes: torch.Tensor, 
                ocr_padding_masks: torch.Tensor, 
                image_sizes: List[tuple]
        ) -> torch.Tensor:
        """
            ocr_boxes: (bs, n_ocr, 4)
            ocr_padding_masks: (bs, 1, 1, n_ocr)
        """
        bs, nq, _ = ocr_boxes.shape
        patch_boxes = torch.zeros((bs, nq, 2), device=ocr_boxes.device)
        for batch in bs:
            image_size = image_sizes[batch]
            for ith in range(nq):
                patch_boxes[batch][ith] = torch.Tensor(self.patch(image_size)).to(ocr_boxes.device)

        dist = torch.zeros((bs, nq, nq)).long().to(ocr_boxes.device)
        for i in range(nq):
            for j in range(i, nq):
                dist[:, i, j] = self.pytha(patch_boxes[:, i], patch_boxes[:, j]).to(ocr_boxes.device)
                dist[:, j, i] = self.pytha(patch_boxes[:, j], patch_boxes[:, i]).to(ocr_boxes.device)
        dist = dist.unsqueeze(-1)# (bs, nq, nq, 1)
        dist = self.dist_embedding(dist) # (bs, nq, nq, d_k)

        q = self.fc_q(ocr_features).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (bs, h, nq, d_k)
        k = self.fc_k(ocr_features).view(bs, nq, self.h, self.d_k).permute(0, 2, 3, 1)  # (bs, h, d_k, nk)
        v = self.fc_v(ocr_features).view(bs, nq, self.h, self.d_v).permute(0, 2, 1, 3)  # (bs, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (bs, h, nq, nq)
        att += ocr_padding_masks
        att = torch.softmax(att + dist, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(bs, nq, self.h * self.d_v)  # (bs, nq, h*d_v)
        out = self.fc_o(out)  # (bs, nq, d_model)

        return out, att
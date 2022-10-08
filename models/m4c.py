import torch
from torch import nn
from torch.nn import functional as F

from .base_unique_transformer import BaseUniqueTransformer
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

import numpy as np
import math
from typing import Dict

class DynamicPointerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.key = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.d_model = config.D_MODEL

    def forward(self, query_inputs, key_inputs, query_attention_mask):
        queries = self.query(query_inputs)
        keys = self.key(key_inputs)
        scores = torch.matmul(queries, keys.transpose((-1, -2))) / math.sqrt(self.d_model)
        scores = scores.masked_fill(query_attention_mask.squeeze(1).squeeze(1), value=-np.inf)

        return scores

@META_ARCHITECTURE.register()
class M4C(BaseUniqueTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)
        self.d_model = config.D_MODEL

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.ocr_det_embedding = build_vision_embedding(config.OCR_DET_EMBEDDING)
        self.ocr_rec_embedding = build_vision_embedding(config.OCR_REC_EMBEDDING)
        self.ocr_text_embedding = build_text_embedding(config.OCR_TEXT_EMBEDDING, vocab)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.self_encoder = build_encoder(config.ENCODER)

        self.dynamic_network = DynamicPointerNetwork(config)
        self.vocab_proj = nn.Linear(config.D_MODEL, len(vocab))

    def forward_region_features(self, features, boxes):
        features, padding_mask = self.region_embedding(features)
        region_feat_tokens = torch.ones((features.shape[0], features.shape[1])).long().to(features.device) * self.vocab.feat_idx
        region_feat_embedded, _ = self.text_embedding(region_feat_tokens)
        features += region_feat_embedded

        boxes, _ = self.box_embedding(boxes)
        region_box_tokens = torch.ones((boxes.shape[0], boxes.shape[1])).long().to(boxes.device) * self.vocab.box_idx
        region_box_embedded, _ = self.text_embedding(region_box_tokens)
        boxes += region_box_embedded

        return features + boxes, padding_mask

    def forward_grid_features(self, features, boxes):
        features, padding_mask = self.grid_embedding(features)
        grid_feat_tokens = torch.ones((features.shape[0], features.shape[1])).long().to(features.device) * self.vocab.feat_idx
        grid_feat_embedded, _ = self.text_embedding(grid_feat_tokens)
        features += grid_feat_embedded
        
        boxes, _ = self.box_embedding(boxes)
        grid_box_tokens = torch.ones((boxes.shape[0], boxes.shape[1])).long().to(boxes.device) * self.vocab.box_idx
        grid_box_embedded, _ = self.text_embedding(grid_box_tokens)
        boxes += grid_box_embedded

        return features + boxes, padding_mask

    def forward_ocr_features(self, ocr_tokens, det_features, rec_features, boxes):
        det_features, padding_mask = self.ocr_det_embedding(det_features)
        ocr_det_tokens = torch.ones((det_features.shape[0], det_features.shape[1])).long().to(det_features.device) * self.vocab.ocr_det_idx
        ocr_det_embedded, _ = self.text_embedding(ocr_det_tokens)
        det_features += ocr_det_embedded

        rec_features, _ = self.ocr_rec_embedding(rec_features)
        ocr_rec_tokens = torch.ones((rec_features.shape[0], rec_features.shape[1])).long().to(rec_features.device) * self.vocab.ocr_rec_idx
        ocr_rec_embedded, _ = self.text_embedding(ocr_rec_tokens)
        rec_features += ocr_rec_embedded

        boxes, _ = self.box_embedding(boxes)
        ocr_box_tokens = torch.ones((boxes.shape[0], boxes.shape[1])).long().to(boxes.device) * self.vocab.box_idx
        ocr_box_embedded, _ = self.text_embedding(ocr_box_tokens)
        boxes += ocr_box_embedded

        ocr_features, _ = self.ocr_text_embedding(ocr_tokens)
        ocr_embedding_tokens = torch.ones((ocr_features.shape[0], ocr_features.shape[1])).long().to(ocr_features.device) * self.vocab.ocr_idx
        ocr_embedded, _ = self.text_embedding(ocr_embedding_tokens)
        ocr_features += ocr_embedded

        return det_features + rec_features + boxes + ocr_features, padding_mask

    def embed_features(self, input_features: Instances):
        region_features = input_features.region_features
        region_boxes = input_features.region_boxes
        region_features, region_padding_mask = self.forward_region_features(region_features, region_boxes)

        grid_features = input_features.grid_features
        grid_boxes = input_features.grid_boxes
        grid_features, grid_padding_mask = self.forward_grid_features(grid_features, grid_boxes)

        ocr_tokens = input_features.ocr_texts
        ocr_det_features = input_features.ocr_det_features
        ocr_rec_features = input_features.ocr_rec_features
        ocr_boxes = input_features.ocr_boxes
        ocr_features, ocr_padding_mask = self.forward_ocr_features(ocr_tokens, ocr_det_features, ocr_rec_features, ocr_boxes)

        question_tokens = input_features.question_tokens
        question_features, question_padding_mask = self.forward_questions(question_tokens)

        vision_features = torch.cat([region_features, grid_features, ocr_features], dim=1)
        vision_padding_mask = torch.cat([region_padding_mask, grid_padding_mask, ocr_padding_mask], dim=-1)

        joint_features = torch.cat([vision_features, question_features], dim=1)
        joint_padding_mask = torch.cat([vision_padding_mask, question_padding_mask], dim=-1)

        joint_features_len = joint_features.shape[1]
        joint_attention_mask = joint_padding_mask.expand((-1, -1, joint_features_len, -1)) # (bs, 1, joint_features_len, joint_features_len)

        maps_tokens_to_features = []
        for batch in range(input_features.batch_size):
            map_tokens_to_features = {}
            ocr_texts = input_features.ocr_texts[batch]
            for idx, token in enumerate(ocr_texts):
                if token not in map_tokens_to_features:
                    map_tokens_to_features[token] = ocr_features[batch, idx]
            maps_tokens_to_features.append(map_tokens_to_features)

        answer_tokens = input_features.answer_tokens
        maps_ids_to_tokens = input_features.maps_ids_to_tokens
        maps_ids_to_features = {}
        for id, token in maps_ids_to_tokens.items():
            maps_ids_to_features[id] = maps_tokens_to_features[token]
        joint_features, (joint_padding_mask, joint_attention_mask) = self.append_answer(joint_features, (joint_padding_mask, joint_attention_mask), 
                                                                                        answer_tokens, maps_ids_to_features)

        return joint_features, (joint_padding_mask, joint_attention_mask)

    def forward_questions(self, question_tokens):
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long().to(question_tokens.device) * self.vocab.question_idx
        question_features, (question_padding_mask, _) = self.text_embedding(question_tokens)
        q_embeded, _ = self.text_embedding(q_tokens)
        question_features += q_embeded

        return question_features, question_padding_mask

    def append_answer(self, joint_features, joint_masks, answer_tokens, maps_ids_to_features):
        answer_features, (answer_padding_mask, answer_sequential_mask) = self.text_embedding(answer_tokens, maps_ids_to_features)
        answer_self_attention_mask = torch.logical_or(answer_padding_mask, answer_sequential_mask) # (bs, 1, answer_len, answer_len)
        a_tokens = torch.ones((answer_tokens.shape[0], answer_tokens.shape[1])).long().to(answer_tokens.device) * self.vocab.answer_idx
        a_embedded, _ = self.text_embedding(a_tokens)
        answer_features += a_embedded

        joint_features_len = joint_features.shape[1]
        answer_len = answer_features.shape[1]
        joint_features = torch.cat([joint_features, answer_features], dim=1)
        joint_padding_mask, joint_attention_mask = joint_masks
        joint_padding_mask = torch.cat([joint_padding_mask, answer_padding_mask], dim=-1)
        # joint features cannot see the answer features
        batch_size = joint_features.shape[0]
        joint_features_mask_answer = torch.ones((batch_size, 1, joint_features_len, answer_len)).bool().to(joint_features.device) # (bs, 1, joint_features_len, answer_len)
        joint_attention_mask = torch.cat([joint_attention_mask, joint_features_mask_answer], dim=-1) # (bs, 1, joint_features_len, joint_features_len + answer_len)
        # answer tokens can attend to all joint features
        answer_attend_joint_features = torch.zeros((batch_size, 1, answer_len, joint_features_len)).bool().to(answer_features.device) # (bs, 1, answer_len, joint_features_len)
        answer_attend_joint_features = torch.cat([answer_attend_joint_features, answer_self_attention_mask], dim=-1) # (bs, 1, answer_len, joint_features_len + answer_len)
        joint_attention_mask = torch.cat([joint_attention_mask, answer_attend_joint_features], dim=-2) # (bs, 1 , joint_features_len + answer_len, joint_features_len + answer_len)

        return joint_features, (joint_padding_mask, joint_attention_mask)

    def forward(self, input_features: Instances):
        region_features_len = input_features.region_features.shape[1]
        grid_features_len = input_features.grid_features.shape[1]
        ocr_tokens_len = input_features.ocr_boxes.shape[1]

        joint_features, (joint_padding_mask, joint_attention_mask) = self.embed_features(input_features)
        ocr_features = joint_features[:, region_features_len+grid_features_len:region_features_len+grid_features_len+ocr_tokens_len]

        encoder_features = self.encoder(Instances(
            features=joint_features,
            features_padding_mask=joint_padding_mask,
            features_attention_mask=joint_attention_mask
        ))

        ocr_features = encoder_features[:, region_features_len+grid_features_len:region_features_len+grid_features_len+ocr_tokens_len]
        ocr_padding_mask = joint_padding_mask[:, region_features_len+grid_features_len:region_features_len+grid_features_len+ocr_tokens_len]

        joint_features_len = joint_features.shape[1]
        answer_features = encoder_features[:, joint_features_len:]
        answer_features = self.vocab_proj(answer_features) # (bs, answer_len, num_vocab)
        ocr_features = self.dynamic_network(answer_features, ocr_features, ocr_padding_mask) # (bs, answer_len, ocr_len)
        out = torch.cat([answer_features, ocr_features], dim=-1) # (bs, answer_len, num_vocab + ocr_len)

        return F.log_softmax(out, dim=-1)

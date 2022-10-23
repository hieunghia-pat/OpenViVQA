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

class DynamicPointerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.key = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.d_model = config.D_MODEL

    def forward(self, query_inputs, key_inputs, query_attention_mask):
        queries = self.query(query_inputs)
        keys = self.key(key_inputs)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(query_attention_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=-np.inf)

        return scores

@META_ARCHITECTURE.register()
class M4C(BaseUniqueTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)
        self.d_model = config.D_MODEL
        self.vocab = vocab
        self.max_len = vocab.max_answer_length
        self.eos_idx = vocab.eos_idx
        self.d_model = config.D_MODEL
        self.max_iter = config.DECODER.MAX_ITER

        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.grid_embedding = build_vision_embedding(config.GRID_EMBEDDING)
        self.box_embedding = build_vision_embedding(config.BOX_EMBEDDING)
        self.ocr_det_embedding = build_vision_embedding(config.OCR_DET_EMBEDDING)
        self.ocr_rec_embedding = build_vision_embedding(config.OCR_REC_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.ocr_embedding = build_text_embedding(config.OCR_TEXT_EMBEDDING, vocab)
        self.dynamic_embedding = build_text_embedding(config.DYNAMIC_EMBEDDING, vocab)

        self.proj_ocr_feat_to_model = nn.Linear(config.D_MODEL*3, config.D_MODEL)

        self.encoder = build_encoder(config.ENCODER)

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

        ocr_word_features, _ = self.ocr_embedding(ocr_tokens)
        ocr_embedding_tokens = torch.ones((ocr_word_features.shape[0], ocr_word_features.shape[1])).long().to(ocr_word_features.device) * self.vocab.ocr_idx
        ocr_embedded, _ = self.text_embedding(ocr_embedding_tokens)
        ocr_word_features += ocr_embedded

        ocr_features = torch.cat([det_features, rec_features, ocr_word_features], dim=-1)
        ocr_features = F.dropout(self.proj_ocr_feat_to_model(ocr_features))
        ocr_features += boxes

        return ocr_features, padding_mask

    def forward_questions(self, question_tokens):
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long().to(question_tokens.device) * self.vocab.question_idx
        question_features, (question_padding_mask, _) = self.text_embedding(question_tokens)
        q_embeded, _ = self.text_embedding(q_tokens)
        question_features += q_embeded

        return question_features, question_padding_mask

    def forward_answer(self, answers, ocr_features):
        answer_features, answer_masks = self.dynamic_embedding(answers, ocr_features)
        a_tokens = torch.ones((answer_features.shape[0], answer_features.shape[1])).long().to(answer_features.device) * self.vocab.answer_idx
        a_embeded, _ = self.text_embedding(a_tokens)
        answer_features += a_embeded

        return answer_features, answer_masks

    def embed_features(self, input_features: Instances):
        region_features = input_features.region_features
        region_boxes = input_features.region_boxes
        region_features, region_padding_mask = self.forward_region_features(region_features, region_boxes)

        grid_features = input_features.grid_features
        grid_boxes = input_features.grid_boxes
        grid_features, grid_padding_mask = self.forward_grid_features(grid_features, grid_boxes)

        ocr_tokens = input_features.ocr_tokens
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

        results = {
            "region_len": region_features.shape[1],
            "grid_len": grid_features.shape[1],
            "ocr_len": ocr_features.shape[1],
            "question_len": question_features.shape[1],
            "joint_features": joint_features,
            "joint_padding_mask": joint_padding_mask,
            "ocr_tokens": ocr_tokens
        }

        return results

    def forward_mmt(self, input_features: Instances):
        embedded_results = self.embed_features(input_features)

        region_len = embedded_results["region_len"]
        grid_len = embedded_results["grid_len"]
        ocr_len = embedded_results["ocr_len"]
        ocr_start = region_len + grid_len
        ocr_end = ocr_start + ocr_len
        
        joint_features = embedded_results["joint_features"]
        joint_padding_mask = embedded_results["joint_padding_mask"]

        answer_tokens = input_features.answer_tokens
        embedded_ocr_features = joint_features[:, ocr_start:ocr_end]
        answer_features, answer_masks = self.forward_answer(answer_tokens, embedded_ocr_features)
        joint_features, (joint_padding_mask, joint_attention_mask) = self.append_answer(joint_features, joint_padding_mask, answer_features, answer_masks)

        encoder_features = self.encoder(
            features=joint_features,
            padding_mask=joint_padding_mask,
            attention_mask=joint_attention_mask
        )

        question_len = embedded_results["question_len"]
        joint_features_len = joint_features.shape[1]
        answer_len = answer_features.shape[1]
        assert joint_features_len == region_len + grid_len + ocr_len + question_len + answer_len

        input_features.answer_features = encoder_features[:, -answer_len:]
        input_features.ocr_features = encoder_features[:, ocr_start:ocr_end]
        input_features.ocr_padding_mask = joint_padding_mask[:, :, :, ocr_start:ocr_end]

        return input_features

    def forward_output(self, input_features):
        answer_features = input_features.answer_features
        ocr_features = input_features.ocr_features
        ocr_padding_mask = input_features.ocr_padding_mask
        vocab_features = self.vocab_proj(answer_features) # (bs, answer_len, num_vocab)
        ocr_features = self.dynamic_network(ocr_features, answer_features, ocr_padding_mask).transpose(-2, -1) # (bs, answer_len, ocr_len)
        out = torch.cat([vocab_features, ocr_features], dim=-1) # (bs, answer_len, num_vocab + ocr_len)

        return out

    def forward(self, input_features: Instances):
        input_features = self.forward_mmt(input_features)
        out = self.forward_output(input_features)
        out = F.log_softmax(out, dim=-1)

        return out
    
    def inference(self, input_features: Instances):
        answer_ids = torch.ones(input_features.batch_size, self.max_len).long().to(self.device) * self.vocab.padding_idx
        answer_ids[:, 0] = self.vocab.bos_idx
        input_features.answer_tokens = answer_ids
        logprob = None
        for step in range(self.max_iter):
            input_features = self.forward_mmt(input_features)
            output = self.forward_output(input_features)
            logprob = F.log_softmax(output, dim=-1)
            answer_ids = output.argmax(dim=-1)
            input_features.answer_tokens[:, 1:] = answer_ids[:, :-1]

        return answer_ids, logprob
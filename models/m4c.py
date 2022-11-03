import torch
from torch import nn
from torch.nn import functional as F

from utils.instance import InstanceList
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

    def forward(self, query_inputs, key_inputs, key_attention_mask):
        queries = self.query(query_inputs)
        keys = self.key(key_inputs)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(key_attention_mask.squeeze(1), value=-np.inf) # there is no head for attention mask

        return scores

@META_ARCHITECTURE.register()
class M4C(nn.Module):
    '''
        Reimplementation of M4C method.
    '''
    def __init__(self, config, vocab):
        super().__init__()

        self.device = torch.device(config.DEVICE)
        self.d_model = config.D_MODEL
        self.vocab = vocab
        self.max_len = vocab.max_answer_length
        self.eos_idx = vocab.eos_idx
        self.d_model = config.D_MODEL

        # embedding for object features
        self.region_embedding = build_vision_embedding(config.REGION_EMBEDDING)
        self.region_box_embedding = build_vision_embedding(config.REGION_BOX_EMBEDDING)

        # embedding for ocr features
        self.ocr_det_embedding = build_vision_embedding(config.OCR_DET_EMBEDDING)
        self.ocr_rec_embedding = build_vision_embedding(config.OCR_REC_EMBEDDING)
        self.ocr_box_embedding = build_vision_embedding(config.OCR_BOX_EMBEDDING)
        self.ocr_token_embedding = build_text_embedding(config.OCR_TEXT_EMBEDDING, vocab)
        self.proj_ocr_features = nn.Linear(config.D_MODEL*3, config.D_MODEL)
        self.norm_ocr_features = nn.LayerNorm(config.D_MODEL)

        # embedding for question
        self.question_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        # embedding for answer
        self.dynamic_embedding = build_text_embedding(config.DYNAMIC_EMBEDDING, vocab)

        # multimodal transformer
        self.encoder = build_encoder(config.ENCODER)

        # output layers
        self.dynamic_network = DynamicPointerNetwork(config)
        self.vocab_proj = nn.Linear(config.D_MODEL, len(vocab))

    def forward_object_features(self, input_features: InstanceList):
        features = input_features.region_features
        features, padding_mask = self.region_embedding(features)

        boxes = input_features.region_boxes
        boxes, _ = self.region_box_embedding(boxes)

        obj_features = features + boxes

        return obj_features, padding_mask

    def forward_ocr_features(self, input_features: InstanceList):
        ocr_det_features = input_features.ocr_det_features
        ocr_det_features, padding_mask = self.ocr_det_embedding(ocr_det_features)

        ocr_rec_features = input_features.ocr_rec_features
        ocr_rec_features, _ = self.ocr_rec_embedding(ocr_rec_features)

        ocr_tokens = input_features.ocr_tokens
        ocr_word_features, _ = self.ocr_token_embedding(ocr_tokens)

        ocr_features = torch.cat([ocr_det_features, ocr_rec_features, ocr_word_features], dim=-1)
        ocr_features = self.proj_ocr_features(ocr_features)
        ocr_features = self.norm_ocr_features(ocr_features)

        ocr_boxes = input_features.ocr_boxes        
        ocr_boxes, _ = self.ocr_box_embedding(ocr_boxes)
        
        ocr_features += ocr_boxes

        return ocr_features, padding_mask

    def forward_questions(self, input_features: InstanceList):
        question_tokens = input_features.question_tokens
        question_features, (question_padding_mask, _) = self.question_embedding(question_tokens)

        return question_features, question_padding_mask

    def forward_mmt(self, input_features: InstanceList):
        # forward input features
        obj_features, obj_padding_mask = self.forward_object_features(input_features)
        ocr_features, ocr_padding_mask = self.forward_ocr_features(input_features)
        question_features, question_padding_mask = self.forward_questions(input_features)

        # forward previous answer tokens
        answer_tokens = input_features.answer_tokens
        answer_features, (answer_padding_mask, answer_sequential_mask) = self.dynamic_embedding(answer_tokens, ocr_features, self.vocab_proj.weight)

        joint_features = torch.cat([obj_features, ocr_features, question_features, answer_features], dim=1)
        joint_padding_mask = torch.cat([obj_padding_mask, ocr_padding_mask, question_padding_mask, answer_padding_mask], dim=-1) # (bs, 1, 1, joint_len)
        # create 3D attention mask for joint attention
        joint_len = joint_features.shape[1]
        joint_attention_mask = joint_padding_mask.repeat(1, 1, joint_len, 1) # (bs, 1, joint_len, joint_len)
        answer_len = answer_features.shape[1]
        joint_attention_mask[:, :,  -answer_len:, -answer_len:] = answer_sequential_mask.squeeze(1)
        
        encoder_outputs = self.encoder(
            features=joint_features,
            padding_mask=joint_padding_mask,
            attention_mask=joint_attention_mask
        )

        # get the offset of features
        obj_len = obj_features.shape[1]
        ocr_len = ocr_features.shape[1]
        question_len = question_features.shape[1]
        ocr_begin = obj_len
        ocr_end = ocr_begin + ocr_len
        answer_begin = obj_len + ocr_len + question_len
        answer_end = answer_begin + answer_len

        ocr_encoder_outputs = encoder_outputs[:, ocr_begin:ocr_end]
        decoder_outputs = encoder_outputs[:, answer_begin:answer_end]

        return decoder_outputs, ocr_encoder_outputs, ocr_padding_mask

    def forward_output(self, input_features: InstanceList):
        decoder_features = input_features.decoder_outputs
        ocr_features = input_features.ocr_encoder_outputs
        ocr_padding_mask = input_features.ocr_padding_mask
        vocab_features = self.vocab_proj(decoder_features) # (bs, answer_len, num_vocab)
        ocr_features = self.dynamic_network(decoder_features, ocr_features, ocr_padding_mask) # (bs, answer_len, ocr_len)
        out = torch.cat([vocab_features, ocr_features], dim=-1) # (bs, answer_len, num_vocab + ocr_len)

        return out

    def forward(self, input_features: InstanceList):
        if self.training:
            decoder_outputs, ocr_encoder_outputs, ocr_padding_mask = self.forward_mmt(input_features)
            input_features.decoder_outputs = decoder_outputs
            input_features.ocr_encoder_outputs = ocr_encoder_outputs
            input_features.ocr_padding_mask = ocr_padding_mask
            output = self.forward_output(input_features)

            results = {"scores": output}
            return results
        else:
            input_features.answer_tokens = torch.ones(input_features.batch_size, self.max_len).long().to(self.device) * self.vocab.padding_idx
            input_features.answer_tokens[:, 0] = self.vocab.bos_idx
            last_ids = torch.zeros((input_features.batch_size, )).to(self.device)
            output = None
            for ith in range(self.max_len):
                decoder_outputs, ocr_encoder_outputs, ocr_padding_mask = self.forward_mmt(input_features)
                input_features.decoder_outputs = decoder_outputs
                input_features.ocr_encoder_outputs = ocr_encoder_outputs
                input_features.ocr_padding_mask = ocr_padding_mask
                output = self.forward_output(input_features)
                answer_ids = output.argmax(dim=-1)
                input_features.answer_tokens[:, 1:] = answer_ids[:, :-1]

                # whether or not to interrupt the decoding process
                last_ids = torch.where(last_ids == self.vocab.eos_idx, last_ids, answer_ids[:, ith])
                if last_ids.mean() == self.vocab.eos_idx:
                    break

            return output
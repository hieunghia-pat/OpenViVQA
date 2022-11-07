import torch
from torch import nn
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import (
    BertConfig,
    BertEncoder,
    BertEmbeddings
)

from utils.instance import InstanceList
from .utils import generate_padding_mask
from builders.text_embedding_builder import build_text_embedding
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

        self.build(config)

    def build(self, config):
        self.build_object_embedding(config)
        self.build_ocr_embedding(config)
        self.build_question_embedding(config)
        self.build_mmt(config)
        self.build_output(config)

    def build_object_embedding(self, config):
        self.linear_obj_feat_to_mmt_in = nn.Linear(config.OBJECT_EMBEDDING.D_FEATURE, config.D_MODEL)

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        self.obj_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_drop = nn.Dropout(config.OBJECT_EMBEDDING.DROPOUT)

    def build_ocr_embedding(self, config):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(config.OCR_EMBEDDING.D_FEATURE, config.D_MODEL)

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        self.ocr_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_text_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_drop = nn.Dropout(config.OCR_EMBEDDING.DROPOUT)

    def build_question_embedding(self, config):
        bert_config = BertConfig(hidden_size=config.TEXT_BERT.HIDDEN_SIZE,
                                    num_hidden_layers=config.TEXT_BERT.NUM_HIDDEN_LAYERS,
                                    num_attention_heads=config.MMT.NUM_ATTENTION_HEADS)
        self.question_embedding = BertEmbeddings(bert_config)
        self.question_encoder = BertEncoder(bert_config)

    def build_mmt(self, config):
        # embedding for answer
        self.dynamic_embedding = build_text_embedding(config.DYNAMIC_EMBEDDING, self.vocab)
        # multimodal transformer
        mmt_config = BertConfig(hidden_size=config.ENCODER.SELF_ATTENTION.D_MODEL,
                                        num_hidden_layers=config.ENCODER.LAYERS,
                                        num_attention_heads=config.ENCODER.SELF_ATTENTION.HEAD)
        self.encoder = BertEncoder(mmt_config)

    def build_output(self, config):
        self.dynamic_network = DynamicPointerNetwork(config)
        self.vocab_proj = nn.Linear(config.D_MODEL, len(self.vocab))

    def forward_object_features(self, input_features: InstanceList):
        features = input_features.region_features
        boxes = input_features.region_boxes
        padding_mask = generate_padding_mask(features, padding_idx=0)

        obj_features = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(features)
        ) + self.obj_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(boxes)
        )
        obj_features = self.obj_drop(obj_features)

        return obj_features, padding_mask

    def forward_ocr_features(self, input_features: InstanceList):
        ocr_det_features = input_features.ocr_det_features
        ocr_det_features = F.normalize(ocr_det_features, dim=-1)

        ocr_rec_features = input_features.ocr_rec_features
        ocr_rec_features = F.normalize(ocr_rec_features, dim=-1)

        ocr_fasttext = input_features.ocr_fasttext_features
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)

        padding_mask = generate_padding_mask(ocr_det_features, padding_idx=0)

        ocr_feat = torch.cat([ocr_det_features, ocr_rec_features, ocr_fasttext], dim=-1)
        ocr_boxes = input_features.ocr_boxes
        ocr_feat_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(ocr_boxes)
        )
        ocr_feat_in = self.ocr_drop(ocr_feat_in)

        return ocr_feat_in, padding_mask

    def forward_questions(self, input_features: InstanceList):
        question_tokens = input_features.question_tokens
        question_features = self.question_embedding(question_tokens)
        
        question_padding_mask = generate_padding_mask(question_tokens, padding_idx=self.vocab.padding_idx)
        attention_mask = question_padding_mask * -10e4
        head_mask = [None] * 12

        question_features = self.question_encoder(
            question_features,
            attention_mask,
            head_mask=head_mask
        )[0]

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
        joint_attention_mask[:, :,  -answer_len:, -answer_len:] = answer_sequential_mask.squeeze()
        # flip the mask, so that invalid attention pairs have -10000
        joint_padding_mask = joint_padding_mask.long() * -10000
        joint_attention_mask = joint_attention_mask.long() * -10000
        head_mask = [None] * 4
        
        encoder_outputs = self.encoder(
            joint_features,
            joint_attention_mask,
            head_mask=head_mask
        )[0]

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
            results = {}
            for ith in range(self.max_len):
                decoder_outputs, ocr_encoder_outputs, ocr_padding_mask = self.forward_mmt(input_features)
                input_features.decoder_outputs = decoder_outputs
                input_features.ocr_encoder_outputs = ocr_encoder_outputs
                input_features.ocr_padding_mask = ocr_padding_mask
                output = self.forward_output(input_features)
                results["scores"] = output
                answer_ids = output.argmax(dim=-1)
                input_features.answer_tokens[:, 1:] = answer_ids[:, :-1]

                # whether or not to interrupt the decoding process
                last_ids = torch.where(last_ids == self.vocab.eos_idx, last_ids, answer_ids[:, ith])
                if last_ids.mean() == self.vocab.eos_idx:
                    break

            return results
import torch
from torch import nn
from torch.nn import functional as F

from .base_transformer import BaseTransformer
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from models.modules.beam_search import BeamSearch
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class UniqueTransformer(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)
        self.vocab = vocab

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)

        self.encoder = build_encoder(config.SELF_ENCODER)
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)

    def embed_features(self, input_features: Instances):
        region_features = input_features.region_features
        region_feat_tokens = torch.ones((region_features.shape[0], region_features.shape[1])).long() * self.vocab.feat_idx
        region_features += self.decoder.word_emb(region_feat_tokens)

        region_boxes = input_features.region_boxes
        region_box_tokens = torch.ones((region_boxes.shape[0], region_boxes.shape[1])).long() * self.vocab.box_idx
        region_boxes += self.decoder.word_emb(region_box_tokens)

        grid_features = input_features.grid_features
        grid_feat_tokens = torch.ones((grid_features.shape[0], region_features.shape[1])).long() * self.vocab.feat_idx
        grid_features += self.decoder.word_emb(grid_feat_tokens)
        
        grid_boxes = input_features.grid_boxes
        grid_box_tokens = torch.ones((grid_boxes.shape[0], region_boxes.shape[1])).long() * self.vocab.box_idx
        grid_boxes += self.decoder.word_emb(grid_box_tokens)

        vision_features = torch.cat([region_features, region_boxes, grid_features, grid_boxes], dim=1)
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        question_features, (question_padding_mask, _) = self.text_embedding(question_tokens)
        q_tokens = torch.ones((question_tokens.shape[0], question_tokens.shape[1])).long * self.vocab.question_idx
        question_features += q_tokens

        joint_features = torch.cat([vision_features, question_features], dim=1)
        joint_attention_mask = torch.cat([vision_padding_mask, question_padding_mask], dim=-1)

        return joint_features, joint_attention_mask

    def append_answer(self, joint_features, joint_attention_mask, answer_tokens):
        answer_features, (answer_padding_mask, answer_sequential_mask) = self.text_embedding(answer_tokens)
        answer_self_attention_mask = torch.logical_or(answer_padding_mask, answer_sequential_mask) # (bs, 1, answer_len, answer_len)
        a_tokens = torch.ones((answer_tokens.shape[0], answer_tokens.shape[1])).long * self.vocab.answer_idx
        answer_features += a_tokens
        
        joint_features = torch.cat([joint_features, answer_features], dim=1)
        answer_len = answer_features.shape[1]
        joint_attention_mask = joint_attention_mask.expand((-1, -1, answer_len, -1))
        joint_attention_mask = torch.cat([joint_attention_mask, answer_self_attention_mask], dim=-1)

        return joint_features, joint_attention_mask

    def forward(self, input_features: Instances):
        joint_features, joint_attention_mask = self.embed_features(input_features)
        joint_features_len = joint_features.shape[1]
        answer_tokens = input_features.answer_tokens
        joint_features, joint_attention_mask = self.append_answer(joint_features, joint_attention_mask, answer_tokens)

        out = self.encoder(Instances(
            features=joint_features,
            features_attention_mask=joint_attention_mask
        ))

        return F.log_softmax(out[:, joint_features_len:], dim=-1)

    def step(self, t, prev_output):
        bs = self.encoder_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.encoder_features.device)
        else:
            it = prev_output

        self.encoder_features, self.encoder_padding_mask = self.append_answer(self.encoder_features, self.encoder_padding_mask, it)
        out = self.encoder(Instances(
            features=self.encoder_features,
            features_attention_mask=self.encoder_padding_mask
        ))

        return F.log_softmax(out[:, self.join_feature_len:], dim=-1)

    def beam_search(self, input_features: Instances, batch_size: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        beam_search = BeamSearch(model=self, max_len=self.max_len, eos_idx=self.eos_idx, beam_size=beam_size, 
                            b_s=batch_size, device=self.device)

        with self.statefulness(batch_size):
            self.encoder_features, self.encoder_padding_mask = self.embed_features(input_features)
            self.join_feature_len = self.encoder_features.shape[1]
            output =  beam_search.apply(out_size, return_probs, **kwargs)

        return output
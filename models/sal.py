import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers.models.t5.modeling_t5 import *
from transformers.generation.utils import *

from utils.logging_utils import Logger
from builders.model_builder import META_ARCHITECTURE
from models.modules.containers import Module
from models.modules.TSS import TextSemanticSeparate
from models.modules.SCP import SpatialCirclePosition
from models.modules.encoders import Encoder
from models.modules.decoders import Decoder
from models.utils import generate_padding_mask

logger = Logger()

@META_ARCHITECTURE.register()
class SAL(Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.config = config
        self.vocab = vocab
        self.max_iter = self.vocab.max_answer_length
        self.device = config.DEVICE

        self.build()

    def build(self):
        self._build_embedding(self.config.EMBEDDING)
        self._build_obj_encoding(self.config.OBJ_EMBEDDING)
        self._build_ocr_encoding(self.config.OCR_EMBEDDING)
        self._build_encoder(self.config.ENCODER)
        self._build_decoder(self.config.DECODER)

    def _build_embedding(self, config):
        self.embedding = nn.Embedding(
            len(self.vocab),
            config.D_MODEL,
            padding_idx=self.vocab.padding_idx
        )

    def _build_obj_encoding(self, config):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            config.OBJ_DIM, 
            config.D_MODEL
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        self.obj_feat_layer_norm = T5LayerNorm(config.D_MODEL)
        self.obj_bbox_layer_norm = T5LayerNorm(config.D_MODEL)
        self.obj_drop = nn.Dropout(config.DROPOUT)

    def _build_ocr_encoding(self, config):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.ocr_dim, 
            self.config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.config.hidden_size)

        self.ocr_feat_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_bbox_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_text_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.dropout_rate)

        # self.tss = TextSemanticSeparate(self.tss_config)
        # self.tss_layer_norm = T5LayerNorm(self.config.hidden_size)

        # self.scp = SpatialCirclePosition(self.scp_config)
        # self.scp_layer_norm = T5LayerNorm(self.config.hidden_size)

    def _build_encoder(self, config):
        self.encoder = Encoder(config)

    def _build_decoder(self, config):
        self.decoder = Decoder(config)
        self.decoder.set_word_emb(self.embedding)
        
    def _forward_question_embedding(self, items):
        # binary mask of valid text (question words) vs padding
        question_tokens = items.question_tokens
        question_padding_mask = generate_padding_mask(question_tokens).to(self.device)
        question_emb = self.shared(question_tokens)

        return question_emb, question_padding_mask

    def _forward_obj_embedding(self, items):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        obj_tag = items.object_list.long()

        obj_emb = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox)
        ) + self.shared(obj_tag)
        obj_emb = self.obj_drop(obj_emb)

        obj_mask = generate_padding_mask(obj_emb, padding_idx=0).to(self.device)

        return obj_emb, obj_mask

    def _forward_ocr_embedding(self, items):
        ocr_rec = items.ocr_rec_features
        ocr_rec = F.normalize(ocr_rec, dim=-1)

        ocr_det = F.normalize(ocr_det, dim=-1)

        ocr_emb = torch.cat(
            [ocr_rec, ocr_det], dim=-1
        )
        ocr_emb = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_emb)
        )

        ocr_bbox = items.ocr_boxes
        ocr_box_emb = self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
        )
        
        ocr_text = items.ocr_tokens
        ocr_text_emb = F.normalize(self.shared(ocr_text), dim=-1)

        ocr_mask = generate_padding_mask(ocr_rec).to(self.device)

        ocr_emb = self.tss_layer_norm(self.tss(
                ocr_emb,
                ocr_box_emb,
                ocr_text_emb
            )
        )

        ocr_emb = self.scp_layer_norm(self.scp(
                ocr_features=ocr_emb,
                ocr_boxes=ocr_bbox, 
                ocr_padding_masks=ocr_mask, 
                image_sizes=items.image_size
            )
        )

        return ocr_emb, ocr_mask

    def _foward_embedding(self, items):
        question_emb, question_mask = self._forward_question_embedding(items)
        obj_emb, obj_mask = self._forward_obj_embedding(items)
        ocr_emb, ocr_mask = self._forward_ocr_embedding(items)
        emb = torch.cat([
            question_emb, 
            obj_emb, 
            ocr_emb], dim=-1)
        mask = torch.cat([
            question_mask,
            obj_mask,
            ocr_mask], dim=-1)
        
        return emb, mask
    
    def forward(self, items):
        embedding, padding_mask = self._foward_embedding(items)
        encoder_features = self.encoder(
            features=embedding,
            padding_mask=padding_mask
        )
        logits = self.decoder(
            answer_tokens=items.answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=padding_mask
        )

        return logits
    
    @torch.no_grad()
    def generate(self, items) -> torch.LongTensor:
        with self.statefulness(batch_size=items.batch_size):
            embedding, padding_mask = self._foward_embedding(items)
            encoder_features = self.encoder(
                features=embedding,
                padding_mask=padding_mask
            )
            self.encoder_features, self.padding_mask = encoder_features, padding_mask
            
            tokens = torch.ones((items.batch_size, 1)).long().to(self.device) * self.vocab.bos_idx
            is_all_eos_tokens = torch.zeros(items.batch_size).long().to(self.device)
            for _ in range(self.max_iter):
                logits = self.decoder(
                    answer_tokens=tokens[:, -1],
                    encoder_features=encoder_features,
                    encoder_attention_mask=padding_mask
                )
                next_token = logits.argmax(dim=-1)
                tokens = torch.cat([tokens, next_token], dim=-1)

                is_all_eos_tokens = torch.where(next_token == self.vocab.eos_idx, True, is_all_eos_tokens)
                # break when all sentences in a batch ending with eos
                if is_all_eos_tokens.mean() == 1:
                    break
            
            return tokens

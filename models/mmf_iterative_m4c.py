import torch
from torch import nn
from torch.nn import functional as F
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertEncoder
)

from .base_transformer import BaseTransformer
from .mmf_regional_m4c import MMF_REGIONAL_M4C, PrevPredEmbeddings, _get_causal_mask, _get_mask
from utils.instance import Instance
from builders.decoder_builder import build_decoder
from builders.model_builder import META_ARCHITECTURE
from models.modules.beam_search import BeamSearch

import numpy as np
import math
from typing import List

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(
        self,
        txt_emb,
        txt_mask,
        obj_emb,
        obj_mask,
        region_emb,
        region_mask,
        ocr_emb,
        ocr_mask
    ):
        encoder_inputs = torch.cat([txt_emb, obj_emb, region_emb, ocr_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, region_mask, ocr_mask], dim=1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        region_max_num = region_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num + region_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output
        }
        return results


@META_ARCHITECTURE.register()
class MMF_IterativeM4C(MMF_REGIONAL_M4C, BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

    def build(self, config):
        self._build_decoder(config)
        super().build()

    def _build_decoder(self, config):
        pass
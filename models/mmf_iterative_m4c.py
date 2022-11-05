import torch
from torch import nn
from pytorch_transformers.modeling_bert import (
    BertConfig,
    BertEncoder
)

from .mmf_regional_m4c import MMF_REGIONAL_M4C, PrevPredEmbeddings, _get_causal_mask, _get_mask
from builders.model_builder import META_ARCHITECTURE

import numpy as np
import math
from typing import List


@META_ARCHITECTURE.register()
class MMF_IterativeM4C(MMF_REGIONAL_M4C):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

    def build(self, config):
        self._build_txt_encoding(config.TEXT_EMBEDDING)
        self._build_obj_encoding(config.OBJECT_EMBEDDING)
        self._build_region_encoding(config.REGION_EMBEDDING)
        self._build_ocr_encoding(config.OCR_EMBEDDING)
        self._build_encoder(config.ENCODER)
        self._build_decoder(config.DECODER)
        self._build_output()

    def _build_encoder(self, config):
        self.encoder_config = BertConfig(
            hidden_size=config.D_MODEL, 
            num_attention_heads=config.HEAD, 
            num_hidden_layers=config.LAYERS,
            hidden_dropout_prob=config.DROPOUT
        )
        self.encoder = BertEncoder(self.encoder_config)

    def _build_decoder(self, config):
        self.decoder_config = BertConfig(
            hidden_size=config.D_MODEL,
            num_attention_heads=config.HEAD, 
            num_hidden_layers=config.LAYERS,
            hidden_dropout_prob=config.DROPOUT
        )
        self.prev_pred_embeddings = PrevPredEmbeddings(self.decoder_config)
        self.decoder = BertEncoder(self.decoder_config)

    def forward(self, items):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(items, fwd_results)
        self._forward_obj_encoding(items, fwd_results)
        self._forward_region_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)
        self._forward_encoder_decoder(items, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_txt_encoding(self, items, fwd_results):
        fwd_results["txt_inds"] = items.question_tokens

        # binary mask of valid text (question words) vs padding
        text_len = (items.question_tokens != self.vocab.padding_idx).sum(dim=-1)
        fwd_results["txt_mask"] = _get_mask(
            text_len, items.question_tokens.size(1)
        )
        # forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"]
        )
        fwd_results["txt_emb"] = self.text_bert_out_linear(text_bert_out)

    def _forward_encoder(self, items, fwd_results):
        txt_emb = fwd_results["txt_emb"]
        txt_mask = fwd_results["txt_mask"]
        obj_emb = fwd_results["obj_mmt_in"]
        obj_mask = fwd_results["obj_mask"]
        region_emb = fwd_results["region_mmt_in"]
        region_mask = fwd_results["region_mask"]
        ocr_emb = fwd_results["ocr_mmt_in"]
        ocr_mask = fwd_results["ocr_mask"]

        encoder_inputs = torch.cat([txt_emb, obj_emb, region_emb, ocr_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, region_mask, ocr_mask], dim=1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        region_max_num = region_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        ocr_begin = txt_max_num + obj_max_num + region_max_num
        ocr_end = ocr_begin + ocr_max_num

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.encoder_config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_encoder_output = encoder_outputs[0]
        mmt_ocr_output = mmt_encoder_output[:, ocr_begin:ocr_end]

        results = {
            "mmt_encoder_output": mmt_encoder_output,
            "mmt_ocr_output": mmt_ocr_output
        }
        return results

    def _forward_decoder(self, items, fwd_results):
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        fixed_ans_emb = fwd_results["fixed_ans_emb"]
        ocr_emb = fwd_results["ocr_mmt_in"]
        prev_inds = fwd_results["prev_inds"]
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)
        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_max_num = dec_emb.shape[1]
        dec_mask = _get_causal_mask(
            dec_max_num, ocr_emb.device
        ).unsqueeze(0).unsqueeze(1)
        # flip the mask, so that invalid attention pairs have -10000.
        dec_mask = (1. - dec_mask) * -10e4
        head_mask = [None] * self.decoder_config.num_hidden_layers

        decoder_outputs = self.decoder(
            dec_emb,
            dec_mask,
            head_mask=head_mask
        )[0]

        fwd_results["mmt_dec_output"] = decoder_outputs

    def _forward_encoder_decoder(self, items, fwd_results):
        self._forward_encoder(items, fwd_results)
        fwd_results["fixed_ans_emb"] = self.classifier.weight
        if self.training:
            fwd_results["prev_inds"] = items.answer_tokens.clone()
            self._forward_decoder(items, fwd_results)
            self._forward_output(items, fwd_results)
        else:
            # greedy decoding at test time
            fwd_results["prev_inds"] = torch.zeros((items.batch_size, 1)).long().to(self.device)
            fwd_results["prev_inds"][:, 0] = self.vocab.bos_idx
            last_ids = torch.zeros((items.batch_size, )).to(self.device)
            for ith in range(self.max_iter):
                self._forward_decoder(items, fwd_results)
                self._forward_output(items, fwd_results)
                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]
                
                # whether or not to interrupt the decoding process
                last_ids = torch.where(last_ids == self.vocab.eos_idx, last_ids, argmax_inds[:, ith])
                if last_ids.mean() == self.vocab.eos_idx:
                    break

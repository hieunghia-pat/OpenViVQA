import torch
from torch import nn
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPreTrainedModel,
)

from .mmf_m4c import MMF_M4C, PrevPredEmbeddings
from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_REGIONAL_M4C(MMF_M4C):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

    def build(self):
        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_region_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()

    def _build_region_encoding(self):
        self.linear_region_feat_to_mmt_in = nn.Linear(
            self.config.REGION_EMBEDDING.D_FEATURE, self.mmt_config.hidden_size
        )

        # region feature: relative bounding box coordinates (4-dim)
        self.linear_region_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.region_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.region_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.region_drop = nn.Dropout(self.config.REGION_EMBEDDING.DROPOUT)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

    def forward(self, items):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(items, fwd_results)
        self._forward_obj_encoding(items, fwd_results)
        self._forward_region_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)
        self._forward_mmt_and_output(items, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_region_encoding(self, items, fwd_results):
        # region appearance feature
        region_feat = items.grid_features
        region_bbox = items.grid_boxes
        region_mmt_in = self.region_feat_layer_norm(
            self.linear_region_feat_to_mmt_in(region_feat)
        ) + self.region_bbox_layer_norm(self.linear_region_bbox_to_mmt_in(region_bbox))
        region_mmt_in = self.obj_drop(region_mmt_in)
        fwd_results["region_mmt_in"] = region_mmt_in

        # binary mask of valid object vs padding
        region_nums = (items.grid_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["region_mask"] = _get_mask(region_nums, region_mmt_in.size(1))

    def _forward_mmt(self, items, fwd_results):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"]
        )
        fwd_results["txt_emb"] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            txt_emb=fwd_results["txt_emb"],
            txt_mask=fwd_results["txt_mask"],
            obj_emb=fwd_results["obj_mmt_in"],
            obj_mask=fwd_results["obj_mask"],
            region_emb=fwd_results["region_mmt_in"],
            region_mask=fwd_results["region_mask"],
            ocr_emb=fwd_results["ocr_mmt_in"],
            ocr_mask=fwd_results["ocr_mask"],
            fixed_ans_emb=self.classifier.weight,
            prev_inds=fwd_results["prev_inds"],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, items, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        mmt_ocr_output = fwd_results["mmt_ocr_output"]
        ocr_mask = fwd_results["ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results["scores"] = scores

    def _forward_mmt_and_output(self, items, fwd_results):
        if self.training:
            fwd_results["prev_inds"] = items.answer_tokens.clone()
            self._forward_mmt(items, fwd_results)
            self._forward_output(items, fwd_results)
        else:
            # fill prev_inds with bos_idx at index 0, and zeros elsewhere
            fwd_results["prev_inds"] = torch.zeros((items.batch_size, self.max_iter)).long().to(self.device)
            fwd_results["prev_inds"][:, 0] = self.vocab.bos_idx

            # greedy decoding at test time
            last_ids = torch.zeros((items.batch_size, )).to(self.device)
            for ith in range(self.max_iter):
                self._forward_mmt(items, fwd_results)
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


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
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
        ocr_mask,
        fixed_ans_emb,
        prev_inds,
    ):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        )
        encoder_inputs = torch.cat([txt_emb, obj_emb, region_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, region_mask, ocr_mask, dec_mask], dim=1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        region_max_num = region_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
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
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
            dec_max_num, encoder_inputs.device
        )

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
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_transformers.modeling_bert import (
    BertConfig,
    BertSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertAttention,
    BertOutput,
    BertEncoder,
    BertPreTrainedModel,
)

from .mmf_m4c import TextBert, MMT, OcrPtrNet, PrevPredEmbeddings, _get_mask, _get_causal_mask
from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE

import math

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_LoRRA(nn.Module):
    """
        This is the modified version of LoRRA method where we replaces the LSTM attention to self-attention of transformer, 
            and adapted decoding module of M4C method to model the OpenViVQA dataset.
        We currently used attention module from pytorch_transformers
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.mmt_config = BertConfig(hidden_size=self.config.MMT.HIDDEN_SIZE,
                                        num_hidden_layers=self.config.MMT.NUM_HIDDEN_LAYERS,
                                        num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        self.vocab = vocab
        self.d_model = self.mmt_config.hidden_size
        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length

        self.build()

    def build(self):
        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(hidden_size=self.config.TEXT_BERT.HIDDEN_SIZE,
                                            num_hidden_layers=self.config.TEXT_BERT.NUM_HIDDEN_LAYERS,
                                            num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        if self.config.TEXT_BERT.LOAD_PRETRAINED:
            self.text_bert = TextBert.from_pretrained(
                self.config.TEXT_BERT.PRETRAINED_NAME, config=self.text_bert_config
            )
        else:
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                self.config.TEXT_BERT.HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.OBJECT_EMBEDDING.D_FEATURE, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.OCR_EMBEDDING.D_FEATURE, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(hidden_size=self.config.OCR_PTR_NET.HIDDEN_SIZE,
                                        query_key_size=self.config.OCR_PTR_NET.QUERY_KEY_SIZE)

        # fixed answer vocabulary scores
        num_choices = len(self.vocab)
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        self.classifier = nn.Linear(self.mmt_config.hidden_size, num_choices)

    def forward(self, items):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(items, fwd_results)
        self._forward_obj_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)
        self._forward_mmt_and_output(items, fwd_results)

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

    def _forward_obj_encoding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = (items.region_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, items, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_feat = items.ocr_fasttext_features
        ocr_feat = F.normalize(ocr_feat, dim=-1)
        assert ocr_feat.size(-1) == 300

        ocr_mmt_in = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_feat))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = (items.ocr_det_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

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

class CrossAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, queries, keys, values, attention_mask, head_mask=None):
        mixed_query_layer = self.query(queries)
        mixed_key_layer = self.key(keys)
        mixed_value_layer = self.value(values)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertCrossAttention(BertAttention):
    def __init__(self, config):
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, src_tensor, tgt_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(src_tensor, tgt_tensor, tgt_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], src_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, src_hidden_states, tgt_hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(src_hidden_states, tgt_hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        
        self.self_att = nn.Identity(config)
        self.spatial_att = BertCrossAttentionLayer(config)
        self.context_att = BertCrossAttentionLayer(config)

        self.decoder = BertEncoder(config)
        
        self.init_weights()

    def forward(
        self,
        txt_emb,
        txt_mask,
        obj_emb,
        obj_mask,
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
        dec_mask = (1 - dec_mask) * -10000.0

        # applying self-attention for question features
        # as M4C used BERT as text encoder hence we do not need attention module anymore
        txt_mask = (1 - txt_mask) * -10000.0
        txt_emb = self.self_att(txt_emb)

        # applying spatial attention between question features and object features
        obj_mask = (1 - obj_mask) * -10000.0
        att_1 = self.spatial_att(txt_emb, obj_emb, obj_mask)

        # applying contextual attention between question feautres and ocr features
        ocr_mask = (1 - ocr_mask) * -10000.0
        att_2 = self.context_att(txt_emb, ocr_emb, ocr_mask)

        # join the self-attention features with spatial attention features
        att = att_1 + att_2
        encoder_inputs = torch.cat([att, dec_emb], dim=1)
        attention_mask = torch.cat([txt_mask, dec_mask], dim=1)
        # create multimodal attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = attention_mask.repeat((1, 1, encoder_inputs.size(1), 1))
        dec_max_num = dec_mask.size(-1)
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(dec_max_num, encoder_inputs.device)

        encoder_outputs = self.decoder(
            encoder_inputs, extended_attention_mask
        )

        mmt_dec_output = encoder_outputs[:, -dec_max_num:]
        results = {
            "mmt_dec_output": mmt_dec_output,
            "mmt_ocr_output": ocr_emb
        }

        return results
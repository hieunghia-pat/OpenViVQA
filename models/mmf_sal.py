import functools
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers.models.t5.modeling_t5 import *

from utils.logging_utils import Logger
from builders.model_builder import META_ARCHITECTURE
from models.modules.TSS import TextSemanticSeparate
from models.modules.SCP import SpatialCirclePosition

logger = Logger()

@META_ARCHITECTURE.register()
class MMF_SAL(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.t5_config = T5Config(
            vocab_size=len(vocab),
            d_model=config.D_MODEL,
            num_layers=config.MMT.NUM_HIDDEN_LAYERS,
            num_heads=config.MMT.NUM_ATTENTION_HEADS,
            num_decoder_layers=config.MMT.NUM_HIDDEN_LAYERS,
            d_ff=config.MMT.D_FF,
            d_kv=config.MMT.D_KV,
            pad_token_id=vocab.padding_idx,
            eos_token_id=vocab.eos_idx
        )
        
        self.config = config

        self.vocab = vocab
        self.d_model = self.t5_config.hidden_size
        self.max_iter = vocab.max_answer_length
        self.device = config.DEVICE

        self.build()
        self.init_weights()

    def init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """Initialize the weights"""
        factor = self.t5_config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, T5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.t5_config.d_model
            key_value_proj_dim = self.t5_config.d_kv
            n_heads = self.t5_config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def build(self):
        self.shared = nn.Embedding(self.t5_config.vocab_size, self.t5_config.d_model)
        
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_encoder()
        self._build_decoder()

        self.vocab_proj = nn.Sequential(
            nn.Linear(self.config.D_MODEL, self.vocab.size()),
            nn.Dropout()
        )

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.OBJECT_EMBEDDING.D_FEATURE, 
            self.t5_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.t5_config.hidden_size)

        self.obj_feat_layer_norm = T5LayerNorm(self.t5_config.hidden_size)
        self.obj_bbox_layer_norm = T5LayerNorm(self.t5_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.OCR_EMBEDDING.D_FEATURE, 
            self.t5_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.t5_config.hidden_size)

        self.ocr_feat_layer_norm = T5LayerNorm(self.t5_config.hidden_size)
        self.ocr_bbox_layer_norm = T5LayerNorm(self.t5_config.hidden_size)
        self.ocr_text_layer_norm = T5LayerNorm(self.t5_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

        self.tss = TextSemanticSeparate(self.config.TSS)
        self.scp = SpatialCirclePosition(self.config.SCP)

    def _build_encoder(self):
        encoder_config = deepcopy(self.t5_config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

    def _build_decoder(self):
        decoder_config = deepcopy(self.t5_config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.t5_config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def forward(self, items):
        if self.training:
            # fwd_results holds intermediate forward pass results
            # TODO possibly replace it with another sample list
            fwd_results = {}
            self._foward_embedding(items, fwd_results)
            self._forward_encoder(items, fwd_results)
            self._forward_decoder(items, fwd_results)

            # only keep scores in the forward pass results
            results = {"scores": fwd_results["scores"]}
            return results
        else:
            fwd_results = {}
            self._foward_embedding(items, fwd_results)
            self._forward_encoder(items, fwd_results)
            self._forward_decoding_step(items, fwd_results)

            # only keep scores in the forward pass results
            results = {"scores": fwd_results["scores"]}
            return results

    def _forward_txt_embedding(self, items, fwd_results):
        fwd_results["txt_inds"] = items.question_tokens

        # binary mask of valid text (question words) vs padding
        text_len = (items.question_tokens != self.vocab.padding_idx).sum(dim=-1)
        fwd_results["txt_mask"] = _get_mask(
            text_len, items.question_tokens.size(1)
        )
        fwd_results["txt_emb"] = self.shared(fwd_results["txt_inds"])

    def _forward_obj_embedding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        obj_tag = items.object_list.long()

        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox)
        ) + self.shared(obj_tag)
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = (items.region_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_embedding(self, items, fwd_results):
        # OCR rec feature (256-dim) extracted from swintextspotter
        ocr_phoc = items.ocr_rec_features
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 256

        # OCR appearance feature, extracted from swintextspotter
        ocr_feat = items.ocr_det_features + items.ocr_rec_features
        ocr_feat = F.normalize(ocr_feat, dim=-1)
        ocr_emb = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        )

        ocr_bbox = items.ocr_boxes
        ocr_box_emb = self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
        )
        
        ocr_text = items.ocr_tokens.long()
        ocr_text_emb = self.shared(ocr_text)

        # binary mask of valid OCR vs padding
        ocr_nums = (ocr_emb.sum(dim=-1) != 0).sum(dim=-1)
        ocr_mask = _get_mask(ocr_nums, ocr_emb.size(1))

        ocr_tss = self.tss(
            ocr_emb,
            ocr_box_emb,
            ocr_text_emb
        )

        ocr_scp, _ = self.scp(
            ocr_features=ocr_tss,
            ocr_boxes=ocr_bbox, 
            ocr_padding_masks=ocr_mask, 
            image_sizes=items.image_size
        )

        fwd_results["ocr_mmt_in"] = ocr_scp
        fwd_results["ocr_mask"] = ocr_mask

    def _foward_embedding(self, items, fwd_results):
        self._forward_txt_embedding(items, fwd_results)
        self._forward_obj_embedding(items, fwd_results)
        self._forward_ocr_embedding(items, fwd_results)
        fwd_results["mmt_in"] = torch.cat([
            fwd_results["txt_emb"], 
            fwd_results["obj_mmt_in"], 
            fwd_results["ocr_mmt_in"]], dim=1)
        fwd_results["mmt_mask"] = torch.cat([
            fwd_results["txt_mask"],
            fwd_results["obj_mask"],
            fwd_results["ocr_mask"]
        ], dim=-1)

    def _forward_encoder(self, items, fwd_results):
        mmt_in = fwd_results["mmt_in"]
        mmt_mask = fwd_results["mmt_mask"]
        encoder_output = self.encoder(
            inputs_embeds = mmt_in,
            encoder_attention_mask=mmt_mask
        )
        fwd_results["mmt_decoder_in"] = encoder_output.last_hidden_state

    def _forward_decoder(self, items, fwd_results):
        fwd_results["prev_inds"] = items.answer_tokens
        bs, seq_len = fwd_results["prev_inds"].shape
        
        answer_lens = (fwd_results["prev_inds"] != self.vocab.padding_idx).sum(dim=-1)
        answer_padding_mask = _get_mask(answer_lens, self.vocab.max_answer_length)[:, None, :]
        answer_casual_mask = _get_causal_mask(bs, seq_len, self.device)
        head_mask = torch.Tensor([1] * self.t5_config.num_hidden_layers)
        mmt_decoder_mask = answer_casual_mask * answer_padding_mask

        mmt_decoder_out = self.decoder(
            input_ids=fwd_results["prev_inds"],
            attention_mask=mmt_decoder_mask,
            encoder_hidden_states=fwd_results["mmt_decoder_in"],
            encoder_attention_mask=fwd_results["mmt_mask"],
            head_mask=head_mask
        ).last_hidden_state
        mmt_decoder_out = self.vocab_proj(mmt_decoder_out)

        fwd_results["scores"] = mmt_decoder_out

    def _forward_decoding_step(self, items, fwd_results):
        bs = items.batch_size
        fwd_results["prev_inds"] = torch.ones((bs, self.vocab.max_answer_length)).long().fill_(self.vocab.padding_idx).to(self.device)
        fwd_results["prev_inds"][:, 0] = self.vocab.bos_idx

        # greedy decoding at test time
        last_ids = torch.zeros((items.batch_size, )).to(self.device)
        for ith in range(1, self.vocab.max_answer_length):
            bs, seq_len = fwd_results["prev_inds"].shape

            answer_lens = (fwd_results["prev_inds"] != self.vocab.padding_idx).sum(dim=-1)
            answer_padding_mask = _get_mask(answer_lens, self.vocab.max_answer_length)[:, None, :]
            answer_casual_mask = _get_causal_mask(bs, seq_len, self.device)
            head_mask = torch.Tensor([1] * self.t5_config.num_hidden_layers)
            mmt_decoder_mask = answer_casual_mask * answer_padding_mask
            mmt_decoder_out = self.decoder(
                input_ids=fwd_results["prev_inds"],
                attention_mask=mmt_decoder_mask,
                encoder_hidden_states=fwd_results["mmt_decoder_in"],
                encoder_attention_mask=fwd_results["mmt_mask"],
                head_mask=head_mask
            ).last_hidden_state
            mmt_decoder_out = self.vocab_proj(mmt_decoder_out)
            
            argmax_inds = mmt_decoder_out.argmax(dim=-1)
            fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]
            
            # whether or not to interrupt the decoding process
            last_ids = torch.where(last_ids == self.vocab.eos_idx, last_ids, argmax_inds[:, ith])
            if last_ids.mean() == self.vocab.eos_idx:
                break

        fwd_results["scores"] = mmt_decoder_out

def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

@functools.lru_cache(maxsize=32)
def _get_causal_mask(bs, seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(bs, seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[:, i, j] = 1.0
    return mask

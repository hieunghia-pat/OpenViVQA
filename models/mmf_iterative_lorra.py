import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertConfig

from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from .utils import generate_padding_mask
from .mmf_m4c import OcrPtrNet, MMT

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_IterativeLoRRA(nn.Module):
    """
        This is the modified version of LoRRA method where we replaces the LSTM attention to self-attention of 
        transformer, and adapted decoding module of M4C method to model the OpenViVQA dataset.
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.mmt_config = BertConfig(hidden_size=self.config.MMT.HIDDEN_SIZE,
                                        num_hidden_layers=self.config.MMT.NUM_HIDDEN_LAYERS,
                                        num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        self.vocab = vocab
        self.d_model = config.D_MODEL
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
        self.txt_embedding = build_text_embedding(self.config.TEXT_EMBEDDING, self.vocab)
        self.txt_norm = nn.LayerNorm(self.d_model)

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.OBJECT_EMBEDDING.D_FEATURE, self.config.D_MODEL
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.config.D_MODEL)

        self.obj_feat_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.OCR_EMBEDDING.D_FEATURE, self.config.D_MODEL
        )

        self.ocr_feat_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

    def _build_mmt(self):
        self.self_attn = build_encoder(self.config.SELF_ATTENTION)
        self.spatial_attn = build_encoder(self.config.SPATIAL_ATTENTION)
        self.context_attn = build_encoder(self.config.CONTEXT_ATTENTION)
        self.mmt = MMT(self.mmt_config)

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(hidden_size=self.config.OCR_PTR_NET.HIDDEN_SIZE,
                                        query_key_size=self.config.OCR_PTR_NET.QUERY_KEY_SIZE)

        # fixed answer vocabulary scores
        num_choices = len(self.vocab)
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        self.classifier = nn.Linear(self.config.D_MODEL, num_choices)

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
        question_tokens = items.question_tokens
        txt_emb, (txt_mask, _) = self.txt_embedding(question_tokens)
        txt_emb = self.txt_norm(txt_emb)

        fwd_results["txt_emb"] = txt_emb
        # binary mask of valid text (question words) vs padding
        fwd_results["txt_mask"] = txt_mask

    def _forward_obj_encoding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_feat_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        )
        obj_feat_in = self.obj_drop(obj_feat_in)
        fwd_results["obj_feat_in"] = obj_feat_in

        # binary mask of valid object vs padding
        fwd_results["obj_mask"] = generate_padding_mask(obj_feat, padding_idx=0)

    def _forward_ocr_encoding(self, items, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_feat = items.ocr_fasttext_features
        ocr_feat = F.normalize(ocr_feat, dim=-1)
        assert ocr_feat.size(-1) == 300

        ocr_mmt_in = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_feat))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        fwd_results["ocr_mask"] = generate_padding_mask(ocr_feat, padding_idx=0)

    def _forward_mmt(self, items, fwd_results):
        txt_emb = fwd_results["txt_emb"]
        txt_padding_mask = fwd_results["txt_mask"]
        self_attn_feat = self.self_attn(
            queries=txt_emb,
            keys=txt_emb,
            values=txt_emb,
            padding_mask=txt_padding_mask,
            attention_mask=txt_padding_mask
        )

        obj_feat_in = fwd_results["obj_feat_in"]
        obj_mask = fwd_results["obj_mask"]
        spatial_attn_feat = self.spatial_attn(
            queries=obj_feat_in,
            keys=self_attn_feat,
            values=self_attn_feat,
            padding_mask=obj_mask,
            attention_mask=txt_padding_mask
        )
        
        ocr_feat_in = fwd_results["ocr_mmt_in"]
        ocr_mask = fwd_results["ocr_mask"]
        context_attn_feat = self.context_attn(
            queries=ocr_feat_in,
            keys=self_attn_feat,
            values=self_attn_feat,
            padding_mask=ocr_mask,
            attention_mask=txt_padding_mask
        )

        mmt_results = self.mmt(
            txt_emb=self_attn_feat,
            txt_mask=fwd_results["txt_mask"].squeeze(2).squeeze(1),
            obj_emb=spatial_attn_feat,
            obj_mask=fwd_results["obj_mask"].squeeze(2).squeeze(1),
            ocr_emb=context_attn_feat,
            ocr_mask=fwd_results["ocr_mask"].squeeze(2).squeeze(1),
            fixed_ans_emb=self.classifier.weight,
            prev_inds=fwd_results["prev_inds"],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, items, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        mmt_ocr_output = fwd_results["mmt_ocr_output"]
        ocr_mask = fwd_results["ocr_mask"].squeeze(2).squeeze(1)

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

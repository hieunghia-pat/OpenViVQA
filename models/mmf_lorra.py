from torch import nn
from torch.nn import functional as F

from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
from builders.attention_builder import build_attention
from builders.text_embedding_builder import build_text_embedding
from .utils import generate_padding_mask

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_LoRRA(nn.Module):
    """
        This is the modified version of LoRRA method where we replaces the LSTM attention to self-attention of 
        transformer
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.d_model = config.D_MODEL
        self.device = config.DEVICE

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
        self.self_attn = build_attention(self.config.SELF_ATTENTION)
        self.spatial_attn = build_attention(self.config.SPATIAL_ATTENTION)
        self.context_attn = build_attention(self.config.CONTEXT_ATTENTION)

    def _build_output(self):
        num_choices = self.vocab.total_answers + self.config.MAX_SCENE_TEXT
        self.classifier = nn.Linear(self.d_model, num_choices)

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
        self_attn_feat, _ = self.self_attn(
            queries=txt_emb,
            keys=txt_emb,
            values=txt_emb,
            padding_mask=txt_padding_mask,
            attention_mask=txt_padding_mask
        )

        obj_feat_in = fwd_results["obj_feat_in"]
        obj_mask = fwd_results["obj_mask"]
        _, spatial_attn_weights = self.spatial_attn(
            queries=obj_feat_in,
            keys=self_attn_feat,
            values=self_attn_feat,
            padding_mask=obj_mask,
            attention_mask=txt_padding_mask
        )
        spatial_attn_weights = spatial_attn_weights.squeeze(1) # (bs, n_obj_feat_in, n_self_attn_feat)
        
        ocr_feat_in = fwd_results["ocr_mmt_in"]
        ocr_mask = fwd_results["ocr_mask"]
        _, context_attn_weights = self.context_attn(
            queries=ocr_feat_in,
            keys=self_attn_feat,
            values=self_attn_feat,
            padding_mask=ocr_mask,
            attention_mask=txt_padding_mask
        )
        context_attn_weights = context_attn_weights.squeeze(1) # (bs, n_ocr_feat_in, n_self_attn_feat)

        attended_spatial_feat = (spatial_attn_weights.unsqueeze(-1) * self_attn_feat.unsqueeze(1)).sum(dim=1)
        attended_context_feat = (context_attn_weights.unsqueeze(-1) * self_attn_feat.unsqueeze(1)).sum(dim=1)
        mmt_feat = attended_spatial_feat + attended_context_feat
        mmt_feat = mmt_feat.sum(dim=1)

        fwd_results["mmt_feat"] = mmt_feat

    def _forward_output(self, items, fwd_results):
        mmt_feat = fwd_results["mmt_feat"]
        output = self.classifier(mmt_feat)

        fwd_results["scores"] = output

    def _forward_mmt_and_output(self, items, fwd_results):
        self._forward_mmt(items, fwd_results)
        self._forward_output(items, fwd_results)

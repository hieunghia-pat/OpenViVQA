import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder

from .mmf_regional_m4c import PrevPredEmbeddings, _get_causal_mask, _get_mask
from .mmf_m4c import OcrPtrNet, TextBert
from builders.model_builder import META_ARCHITECTURE
from utils.logging_utils import setup_logger

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_IterativeM4C(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.D_MODEL
        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length

        self.build(config)

    def build(self, config):
        self._build_txt_encoding(config.TEXT_BERT)
        self._build_obj_encoding(config.OBJECT_EMBEDDING)
        self._build_region_encoding(config.REGION_EMBEDDING)
        self._build_ocr_encoding(config.OCR_EMBEDDING)
        self._build_encoder(config.ENCODER)
        self._build_decoder(config.DECODER)
        self._build_output(config.OCR_PTR_NET)

    def _build_txt_encoding(self, config):
        text_bert_config = BertConfig(hidden_size=config.HIDDEN_SIZE,
                                            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
                                            num_attention_heads=config.NUM_ATTENTION_HEADS)
        self.text_bert = TextBert(text_bert_config)
        self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self, config):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            config.D_FEATURE, config.D_MODEL
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        self.obj_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.obj_drop = nn.Dropout(config.DROPOUT)

    def _build_region_encoding(self, config):
        self.linear_region_feat_to_mmt_in = nn.Linear(
            config.D_FEATURE, config.D_MODEL
        )

        # region feature: relative bounding box coordinates (4-dim)
        self.linear_region_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        self.region_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.region_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.region_drop = nn.Dropout(config.DROPOUT)

    def _build_ocr_encoding(self, config):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            config.D_FEATURE, config.D_MODEL
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, config.D_MODEL)

        # OCR word embedding features
        # self.ocr_word_embedding = build_word_embedding(self.config.OCR_TEXT_EMBEDDING)

        self.ocr_feat_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_bbox_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_text_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.ocr_drop = nn.Dropout(config.DROPOUT)

    def _build_output(self, config):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(hidden_size=config.HIDDEN_SIZE,
                                        query_key_size=config.QUERY_KEY_SIZE)

        # fixed answer vocabulary scores
        num_choices = len(self.vocab)
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        self.classifier = nn.Linear(self.d_model, num_choices)

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
            add_cross_attention=True,
            is_decoder=True,
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

    def _forward_ocr_encoding(self, items, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = items.ocr_fasttext_features
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR rec feature (256-dim), replace the OCR PHOC features, extracted from swintextspotter
        ocr_phoc = items.ocr_rec_features
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 256

        # OCR appearance feature, extracted from swintextspotter
        ocr_fc = items.ocr_det_features
        ocr_fc = F.normalize(ocr_fc, dim=-1)

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc], dim=-1
        )
        ocr_bbox = items.ocr_boxes
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = (items.ocr_det_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

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

        fwd_results["mmt_encoder_output"] = mmt_encoder_output
        fwd_results["mmt_encoder_mask"] = extended_attention_mask
        fwd_results["mmt_ocr_output"] = mmt_ocr_output

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
            hidden_states=dec_emb,
            attention_mask=dec_mask,
            encoder_hidden_states=fwd_results["mmt_encoder_output"],
            encoder_attention_mask=fwd_results["mmt_encoder_mask"],
            head_mask=head_mask
        )[0]

        fwd_results["mmt_dec_output"] = decoder_outputs

    def _forward_output(self, items, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        mmt_ocr_output = fwd_results["mmt_ocr_output"]
        ocr_mask = fwd_results["ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results["scores"] = scores

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

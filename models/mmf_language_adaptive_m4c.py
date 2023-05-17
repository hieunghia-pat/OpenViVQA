import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertConfig, 
    BertEncoder,
    BertPreTrainedModel
)
from transformers import AutoModel, AutoTokenizer

from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
from .mmf_m4c import OcrPtrNet, MMT

logger = setup_logger()

@META_ARCHITECTURE.register()
class MMF_LanguageAdaptiveM4C(nn.Module):
    '''
        This is the original version of M4C method copied directly from https://github.com/ronghanghu/mmf
    '''
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
        self.text_bert_config = BertConfig(hidden_size=self.config.TEXT_BERT.HIDDEN_SIZE,
                                            num_hidden_layers=self.config.TEXT_BERT.NUM_HIDDEN_LAYERS,
                                            num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        self.text_bert = PretrainedAdaptiveTextBert(self.text_bert_config, self.config.TEXT_BERT.PRETRAINED_NAME, self.config.TEXT_BERT.D_LANGUAGE)

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

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        # OCR word embedding features
        # self.ocr_word_embedding = build_word_embedding(self.config.OCR_TEXT_EMBEDDING)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_text_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
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
        question_str = items.question
        txt_emb, txt_mask = self.text_bert(question_str)
        fwd_results["txt_emb"] = txt_emb
        fwd_results["txt_mask"] = txt_mask

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

    def _forward_mmt(self, items, fwd_results):
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

class PretrainedAdaptiveTextBert(BertPreTrainedModel):
    def __init__(self, config, pretrained_name: str, pretrained_dim: int):
        super().__init__(config)

        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        # embedding layer is the pretrained language model
        self.embedding = AutoModel.from_pretrained(pretrained_name)
        # freeze the pretrained language model
        for param in self.embedding.parameters():
            param.require_grad = False

        TEXT_BERT_HIDDEN_SIZE = pretrained_dim
        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

        # fine tuning layer
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_str):
        tokenizer_outputs = self.pretrained_tokenizer(txt_str, return_tensors='pt', padding=True)
        txt_inds = tokenizer_outputs["input_ids"].to(self.device)
        attention_mask = tokenizer_outputs["attention_mask"].to(self.device)

        encoder_inputs = self.embedding(txt_inds).last_hidden_state
        encoder_inputs = self.text_bert_out_linear(encoder_inputs)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )[0]

        return encoder_outputs, attention_mask

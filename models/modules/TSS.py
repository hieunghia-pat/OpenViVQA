import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer

from typing import List

class TextSemanticSeparate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        t5_model = AutoModel.from_pretrained(config.PRETRAINED_EMBEDDING)
        self.ocr_embedding = t5_model.shared
        self.ocr_embedding.requires_grad = False

        self.t5_tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_EMBEDDING)
        # <pad> will be used as <context>
        self.context_token = self.t5_tokenizer.pad_token

        self.device = torch.device(config.DEVICE)

    def forward(self,
                obj_emb: torch.Tensor,
                obj_box_emb: torch.Tensor,
                ocr_emb: torch.Tensor,
                ocr_box_emb: torch.Tensor,
                ocr_text: List[List[str]]) -> torch.Tensor:
        """
            obj_emb: (bs, n_obj, d_model)
            obj_box_emb: (bs, n_obj, d_model)
            ocr_emb: (bs, n_ocr, d_model)
            ocr_box_emb: (bs, n_ocr, d_model)
            ocr_text: (bs, l, 1)
        """

        # get the T5 embedding features for OCR texts
        bs = obj_emb.shape[0]
        for batch in range(bs):
            for idx, _ in enumerate(ocr_text[batch]):
                ocr_text[batch].insert(idx+1, self.context_token)

        ocr_text = [" ".join(text) for text in ocr_text]
        ocr_text_emb = self.t5_tokenizer(ocr_text, return_tensors="pt")["input_ids"]
        
        # extending embedded features in order to add them to context tokens
        extended_obj_emb = torch.zeros_like(ocr_text_emb).to(self.device)
        extended_obj_emb[:, 0::2] = obj_emb
        extended_obj_emb[:, 1::2] = obj_emb

        extended_obj_box_emb = torch.zeros_like(ocr_text_emb).to(self.device)
        extended_obj_box_emb[:, 0::2] = obj_box_emb
        extended_obj_box_emb[:, 1::2] = obj_box_emb

        extended_ocr_emb = torch.zeros_like(ocr_text_emb).to(self.device)
        extended_ocr_emb[:, 0::2] = ocr_emb
        extended_ocr_emb[:, 1::2] = ocr_emb

        extended_ocr_box_emb = torch.zeros_like(ocr_text_emb).to(self.device)
        extended_ocr_box_emb[:, 0::2] = ocr_box_emb
        extended_ocr_box_emb[:, 1::2] = ocr_box_emb

        tss_features = ocr_text_emb + extended_obj_emb + extended_obj_box_emb \
                        + extended_ocr_emb + extended_ocr_box_emb
        
        return tss_features

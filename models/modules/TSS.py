import torch
from torch import nn

class TextSemanticSeparate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.device = torch.device(config.DEVICE)
        self.context_emb = nn.Parameter(nn.init.xavier_uniform(
            torch.zeros(1, config.D_MODEL)
        ))

    def forward(self,
                ocr_emb: torch.Tensor,
                ocr_box_emb: torch.Tensor,
                ocr_text_emb: torch.Tensor
            ) -> torch.Tensor:
        """
            ocr_emb: (bs, n_ocr, d_model)
            ocr_box_emb: (bs, n_ocr, d_model)
            ocr_text_emb: (bs, n_ocr, d_model)
        """

        extended_ocr_emb = torch.zeros_like(ocr_emb).to(self.device)
        extended_ocr_emb[:, 0::2] = ocr_emb
        extended_ocr_emb[:, 1::2] = ocr_emb

        extended_ocr_box_emb = torch.zeros_like(ocr_box_emb).to(self.device)
        extended_ocr_box_emb[:, 0::2] = ocr_box_emb
        extended_ocr_box_emb[:, 1::2] = ocr_box_emb

        extended_ocr_text_emb = torch.zeros_like(ocr_text_emb).to(self.device)
        extended_ocr_text_emb[:, 0::2] = ocr_text_emb
        extended_ocr_text_emb[:, 1::2] = self.context_emb

        tss_features = extended_ocr_emb + extended_ocr_box_emb + extended_ocr_text_emb

        return tss_features[:, 0::2] + tss_features[:, 1::2]

import torch
from torch import nn
from models.captioning_model import CaptioningModel
from models.modules.embeddings import SinusoidPositionalEmbedding

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, use_img_pos=False):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.use_img_pos = use_img_pos
        if self.use_img_pos:
            self.sinusoid_pos_embedding = SinusoidPositionalEmbedding(encoder.d_model, normalize=True)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input, tokens, boxes=None, grid_size=None):
        pos_emb = self.sinusoid_pos_embedding(input) if self.use_img_pos else None
        enc_output, mask_enc = self.encoder(input, boxes, grid_size, positional_emb=pos_emb)
        dec_output = self.decoder(tokens, enc_output, mask_encoder=mask_enc, positional_emb=pos_emb)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, boxes=None, grid_size=None, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            pos_emb = self.sinusoid_pos_embedding(visual) if self.use_img_pos else None
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual, boxes, grid_size, positional_emb=pos_emb)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, mask_encoder=self.mask_enc, positional_emb=pos_emb)
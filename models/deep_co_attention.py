from torch import nn

class DeepCoAttention(nn.Module):
    def __init__(self, d_model, dff, nheads, nlayers, dropout=0.5):
        super(DeepCoAttention, self).__init__()

        self_attention = nn.TransformerEncoderLayer(d_model, nheads, dff, dropout, batch_first=True)
        guided_attention = nn.TransformerDecoderLayer(d_model, nheads, dff, dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            self_attention,
            nlayers
        )
        
        self.decoder = nn.TransformerDecoder(
            guided_attention,
            nlayers
        )

    def forward(self, v, q, attn_mask, key_padding_mask):
        q_encoded = self.encoder(src=q, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        v_encoded = self.decoder(tgt=v, memory=q_encoded)

        return v_encoded, q_encoded

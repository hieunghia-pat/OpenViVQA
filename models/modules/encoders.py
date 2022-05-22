import torch
from torch import nn
from torch.nn import functional as F
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from models.utils import generate_padding_mask, generate_sequential_mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, boxes=None, grid_size=None, positional_emb=None, attention_mask=None, attention_weights=None):
        if positional_emb is not None:
            queries += positional_emb
            keys += positional_emb
        att = self.mhatt(queries, keys, values, boxes=boxes, grid_size=grid_size, attention_mask=attention_mask, attention_weights=attention_weights)
        ff = self.pwff(att)
        return ff

class GuidedEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(GuidedEncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.self_mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.guided_mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, boxes=None, grid_size=None, positional_emb=None, 
                    mask_pad=None, self_attention_mask=None, guided_attention_mask=None, 
                    self_attention_weights=None, guided_attention_weights=None):
        if positional_emb is not None:
            queries += positional_emb

        queries = self.self_mhatt(queries, queries, queries, boxes=boxes, grid_size=grid_size, 
                                    attention_mask=self_attention_mask, attention_weights=self_attention_weights)
        queries = queries.masked_fill(mask_pad, value=0)

        guided_att = self.guided_mhatt(queries, keys, values, boxes=boxes, grid_size=grid_size, 
                                    attention_mask=guided_attention_mask, attention_weights=guided_attention_weights)
        guided_att = guided_att.mask_fill(mask_pad, value=0)

        ff = self.pwff(guided_att)
        return ff

class Encoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(Encoder, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, boxes=None, grid_size=None, positional_emb=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, out, out, boxes, grid_size, positional_emb, attention_mask, attention_weights)

        return out, attention_mask

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, boxes=None, grid_size=None, positional_emb=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        # blank features are added by zero tensors
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, out, out, boxes, grid_size, positional_emb, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, dim=1)
        return outs, attention_mask

class GuidedEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([GuidedEncoder(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, visuals, questions, boxes=None, grid_size=None, positional_emb=None, mask_questions=None):
        seq_len = visuals.shape[1]
        mask_visuals = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        mask_self_attention = generate_sequential_mask(seq_len).to(questions.device)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = torch.logical_or(mask_self_attention, mask_visuals.unsqueeze(1).unsqueeze(1))

        visuals = self.layer_norm(visuals)
        for layer in self.layers:
            visuals = layer(visuals, questions, questions, boxes=boxes, grid_size=grid_size, 
                            positiona_emb=positional_emb, mask_pad=mask_visuals, 
                            self_attention_mask=mask_self_attention, guided_attention_mask=mask_questions)

        return visuals, mask_visuals
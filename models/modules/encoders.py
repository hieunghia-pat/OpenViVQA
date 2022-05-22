import torch
from torch import nn
from torch.nn import functional as F
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention

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

    def forward(self, queries, keys, values, boxes=None, attention_mask=None):
        att = self.mhatt(queries, keys, values, boxes=boxes, attention_mask=attention_mask)
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

    def forward(self, queries, keys, values, boxes=None,
                self_attention_mask=None, guided_attention_mask=None):

        queries = self.self_mhatt(queries, queries, queries, boxes=boxes,
                                    attention_mask=self_attention_mask)

        guided_att = self.guided_mhatt(queries, keys, values, boxes=boxes,
                                    attention_mask=guided_attention_mask)

        ff = self.pwff(guided_att)
        return ff

class Encoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
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

    def forward(self, visuals):
        features = visuals.features
        padding_mask = visuals.feature_padding_masks
        pos_embeddings = visuals.feature_pos_embeddings
        boxes = visuals.boxes
        
        out = F.relu(self.fc(features))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = out + pos_embeddings
            out = layer(out, out, out, boxes, padding_mask)

        return out, padding_mask

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
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

    def forward(self, visuals):
        features = visuals.features
        padding_mask = visuals.feature_padding_masks
        pos_embeddings = visuals.feature_pos_embeddings
        boxes = visuals.boxes

        outs = []
        out = F.relu(self.fc(features))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = out + pos_embeddings
            out = layer(out, out, out, boxes, pos_embeddings, padding_mask)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, dim=1)
        return outs, padding_mask

class GuidedEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([GuidedEncoder(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

    def forward(self, visuals, linguistics):
        
        features = visuals.features
        feature_padding_masks = visuals.feature_padding_masks
        feature_pos_embeddings = visuals.feature_pos_embeddings
        boxes = visuals.boxes

        questions = linguistics.questions
        question_attention_masks = linguistics.question_attention_masks
        question_pos_embeddings = linguistics.question_pos_embeddings
        questions = questions + question_pos_embeddings

        features = self.layer_norm(features)
        for layer in self.layers:
            features = features + feature_pos_embeddings
            features = layer(features, questions, questions, boxes=boxes,
                            self_attention_mask=question_attention_masks, guided_attention_mask=feature_padding_masks)

        return features, feature_padding_masks
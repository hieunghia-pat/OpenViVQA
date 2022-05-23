from torch import nn
from torch.nn import functional as F
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from models.modules.embeddings import SinusoidPositionalEmbedding

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

    def forward(self, queries, keys, values, attention_mask=None):
        att = self.mhatt(queries, keys, values, attention_mask=attention_mask)
        ff = self.pwff(att)
        return ff

class GeometricEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(GeometricEncoderLayer, self).__init__()
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
                 use_aoa=False, self_attention_module=None, self_attention_module_kwargs=None,
                 guided_attention_module=None, guided_attention_module_kwargs=None):
        super(GuidedEncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.self_mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=self_attention_module,
                                        attention_module_kwargs=self_attention_module_kwargs)
        self.guided_mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=guided_attention_module,
                                        attention_module_kwargs=guided_attention_module_kwargs)
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

        self.pos_embedding = SinusoidPositionalEmbedding(d_model // 2, normalize=True)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

    def forward(self, general_features):
        features = general_features.features
        padding_masks = general_features.masks
        pos_embeddings = self.pos_embedding(features)
        
        out = F.relu(self.fc(features))
        out = self.dropout(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = out + pos_embeddings
            out = layer(out, out, out, padding_masks)

        return out

class GeometricEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, use_aoa=False, attention_module=None, attention_module_kwargs=None):
        super(GeometricEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(d_model // 2, normalize=True)
        self.box_embedding = nn.Linear(in_features=4, out_features=d_model)
        
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
        padding_masks = visuals.masks
        pos_embeddings = self.pos_embedding(features)
        boxes = visuals.boxes
        
        out = F.relu(self.fc(features))
        out = self.dropout(out)
        out = self.layer_norm(out)
        boxes = self.box_embedding(boxes)
        for layer in self.layers:
            out = out + pos_embeddings
            out = layer(out, out, out, boxes=boxes, attention_mask=padding_masks)

        return out

class GuidedEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, 
                    dropout=.1, identity_map_reordering=False, use_aoa=False,
                    self_attention_module=None, self_attention_module_kwargs=None,
                    guided_attention_module=None, guided_attention_module_kwargs=None):
        super(GuidedEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(d_model // 2, normalize=True)
        self.box_embedding = nn.Linear(in_features=4, out_features=d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers = nn.ModuleList([GuidedEncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  self_attention_module=self_attention_module,
                                                  self_attention_module_kwargs=self_attention_module_kwargs,
                                                  guided_attention_module=guided_attention_module,
                                                  guided_attention_module_kwargs=guided_attention_module_kwargs)
                                     for _ in range(N)])

    def forward(self, visuals, linguistics):
        
        visual_features = visuals.features
        visual_feature_padding_masks = visuals.masks
        visual_feature_pos_embeddings = self.pos_embedding(visual_features)

        linguistic_features = linguistics.features
        linguistic_feature_attention_masks = linguistics.masks
        linguistic_feature_pos_embeddings = self.pos_embedding(linguistic_features)

        visual_features = self.layer_norm(visual_features)
        linguistic_features = linguistic_features + linguistic_feature_pos_embeddings
        for layer in self.layers:
            visual_features = visual_features + visual_feature_pos_embeddings
            visual_features = layer(visual_features, linguistic_features, linguistic_features,
                            self_attention_mask=linguistic_feature_attention_masks, guided_attention_mask=visual_feature_padding_masks)

        return visual_features
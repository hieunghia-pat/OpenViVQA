import torch
from torch import nn

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from builders.encoder_builder import META_ENCODER

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)

        return ff

class CrossModalityEncoderLayer(nn.Module):
    def __init__(self, config):
        super(CrossModalityEncoderLayer, self).__init__()
        
        # cross-attention modules
        self.vision_language_mhattn = MultiHeadAttention(config.VISION_LANGUAGE_ATTENTION)
        self.language_vision_mhattn = MultiHeadAttention(config.LANGUAGE_VISION_ATTENTION)

        # self-attention modules
        self.vision_mhattn = MultiHeadAttention(config.VISION_SELF_ATTENTION)
        self.language_mhattn = MultiHeadAttention(config.LANGUAGE_SELF_ATTENTION)

        # pff
        self.vision_pff = PositionWiseFeedForward(config.VISION_SELF_ATTENTION)
        self.language_pff = PositionWiseFeedForward(config.LANGUAGE_SELF_ATTENTION)

    def forward(self, vision_features, vision_padding_mask, language_features, language_padding_mask, **kwargs):
        # perform cross-attention
        vision_attn = self.vision_language_mhattn(
            queries=vision_features,
            keys=language_features,
            values=language_features,
            attention_mask=language_padding_mask,
            **kwargs
        )
        language_attn = self.language_vision_mhattn(
            queries=language_features,
            keys=vision_features,
            values=vision_features,
            attention_mask=vision_padding_mask
        )

        # perform self-attention
        vision_attn = self.vision_mhattn(
            queries=vision_features,
            keys=vision_features,
            values=vision_features,
            attention_mask=vision_padding_mask,
            **kwargs
        )
        language_attn = self.language_mhattn(
            queries=language_features,
            keys=language_features,
            values=language_features,
            attention_mask=language_padding_mask
        )

        # perform pff
        vision_attn = self.vision_pff(vision_attn)
        language_attn = self.language_pff(language_attn)

        return vision_attn, language_attn

class GuidedEncoderLayer(nn.Module):
    def __init__(self, config):
        super(GuidedEncoderLayer, self).__init__()
        self.self_mhatt = MultiHeadAttention(config)
        self.guided_mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, self_attention_mask, guided_attention_mask, **kwargs):
        self_att = self.self_mhatt(
                                    queries=queries,
                                    keys=queries, 
                                    values=queries,
                                    attention_mask=self_attention_mask,
                                    **kwargs
                                )
        guided_att = self.guided_mhatt(
                                        queries=self_att, 
                                        keys=keys, 
                                        values=values,
                                        attention_mask=guided_attention_mask,
                                        **kwargs
                                    )

        ff = self.pwff(guided_att)

        return ff

@META_ENCODER.register()
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        out = self.layer_norm(features) + self.pos_embedding(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_mask)

        return out

@META_ENCODER.register()
class GeometricEncoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, features: torch.Tensor, boxes: torch.Tensor, padding_mask: torch.Tensor):    
        out = self.layer_norm(features) + self.pos_embedding(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, boxes=boxes, attention_mask=padding_mask)

        return out

@META_ENCODER.register()
class GuidedAttentionEncoder(nn.Module):
    '''
        This module is designed inspired from Deep Modular Co-Attention Network (https://arxiv.org/pdf/1906.10770.pdf).
    '''
    def __init__(self, config):
        super(GuidedAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        self.guided_attn_layers = nn.ModuleList([GuidedEncoderLayer(config.GUIDED_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        out = self.layer_norm(vision_features) + self.pos_embedding(vision_features)
        for guided_attn_layer in self.guided_attn_layers:
            out = guided_attn_layer(
                queries=out,
                keys=language_features,
                values=language_features,
                self_attention_mask=vision_padding_mask,
                guided_attention_mask=language_padding_mask
            )

        return out

@META_ENCODER.register()
class CoAttentionEncoder(nn.Module):
    '''
        This module is designed inspired from ViLBERT (https://arxiv.org/pdf/1908.02265.pdf).
    '''
    def __init__(self, config):
        super(CoAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.vision_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.language_layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        # cross-attention layers
        self.vision_language_attn_layers = nn.ModuleList([EncoderLayer(config.VISION_LANGUAGE_ATTENTION) for _ in range(config.LAYERS)])
        self.language_vision_attn_layers = nn.ModuleList([EncoderLayer(config.LANGUAGE_VISION_ATTENTION) for _ in range(config.LAYERS)])

        # self-attention layers
        self.vision_self_attn_layers = nn.ModuleList([EncoderLayer(config.VISION_SELF_ATTENTION) for _ in range(config.LAYERS)])
        self.language_self_attn_layers = nn.ModuleList([EncoderLayer(config.LANGUAGE_SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        vision_features = self.vision_layer_norm(vision_features) + self.pos_embedding(vision_features)
        language_features = self.language_layer_norm(language_features) + self.pos_embedding(language_features)
        for layers in zip(self.vision_language_attn_layers, 
                            self.language_vision_attn_layers, 
                            self.vision_self_attn_layers, 
                            self.language_self_attn_layers):
            vision_language_attn_layer, language_vision_attn_layer, vision_self_attn_layer, language_self_attn_layer = layers
            # performing cross-attention
            vision_features = vision_language_attn_layer(
                queries=vision_features,
                keys=language_features,
                values=language_features,
                attention_mask=language_padding_mask
            )
            language_features = language_vision_attn_layer(
                queries=language_features,
                keys=vision_features,
                values=vision_features,
                attention_mask=vision_padding_mask
            )
            # performing self-attention
            vision_features = vision_self_attn_layer(
                queries=vision_features,
                keys=vision_features,
                values=vision_features,
                attention_mask=vision_padding_mask
            )
            language_features = language_self_attn_layer(
                queries=language_features,
                keys=language_features,
                values=language_features,
                attention_mask=language_padding_mask
            )

        return vision_features, language_features

@META_ENCODER.register()
class CrossModalityEncoder(nn.Module):
    '''
        This module is designed inspired from LXMERT (https://arxiv.org/pdf/1908.07490.pdf).
    '''
    def __init__(self, config):
        super(CrossModalityEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.vision_layer_norm = nn.LayerNorm(config.D_MODEL)
        self.language_layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([CrossModalityEncoderLayer(config) for _ in range(config.LAYERS)])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        vision_features = self.vision_layer_norm(vision_features) + self.pos_embedding(vision_features)
        language_features = self.language_layer_norm(language_features) + self.pos_embedding(language_features)
        for layer in self.layers:
            vision_features, language_features = layer(
                vision_features=vision_features,
                vision_padding_mask=vision_padding_mask,
                language_features=language_features,
                language_padding_mask=language_padding_mask
            )

        return vision_features, language_features

from torch import nn

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention
from builders.encoder_builder import META_ENCODER
from utils.instances import Instances

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
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
        self.vision_pff = PositionWiseFeedForward(config.VISION_PFF)
        self.language_pff = PositionWiseFeedForward(config.LANGUAGE_PFF)

    def forward(self, vision_features, vision_padding_mask, language_features, language_padding_mask, **kwargs):
        # perform cross-attention
        vision_attn = self.vision_language_mhattn(
            queries=vision_features,
            keys=language_features,
            values=language_features,
            attention_mask=vision_padding_mask,
            **kwargs
        )
        language_attn = self.language_mhattn(
            queries=language_features,
            keys=vision_features,
            values=vision_features,
            attention_mask=language_padding_mask
        )

        # perform self-attention
        vision_attn = self.vision_mhattn(
            queries=vision_features,
            keys=vision_features,
            values=vision_features,
            attention_mask=vision_features,
            **kwargs
        )
        language_attn = self.language_mhattn(
            queries=language_features,
            keys=language_features,
            values=language_features,
            attention_mask=language_features
        )

        # perform pff
        vision_attn = self.vision_pff(vision_attn)
        language_attn = self.language_pff(language_attn)

        return vision_attn, language_attn

@META_ENCODER.register()
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        features = input_features.features
        padding_mask = input_features.features_padding_mask
        
        out = self.layer_norm(features)
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_mask)

        return out

@META_ENCODER.register()
class GeometricEncoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        features = input_features.features
        boxes = input_features.boxes
        padding_mask = input_features.features_padding_mask
        
        out = self.layer_norm(features)
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

        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        self.self_attn_layers = nn.ModuleList([EncoderLayer(config.SELF_ATTENTION)] for _ in range(config.LAYERS))
        self.guided_attn_layers = nn.ModuleList([EncoderLayer(config.GUIDED_ATTENTION)] for _ in range(config.LAYERS))

    def forward(self, input_features: Instances):
        vision_features = input_features.vision_features
        boxes = input_features.boxes
        vision_padding_mask = input_features.vision_padding_mask

        language_features = input_features.language_padding_mask
        language_padding_mask = input_features.language_padding_mask

        for self_attn_layer, guided_attn_layer in zip(self.self_attn_layers, self.guided_attn_layers):
            # pass to the self-attention layer
            vision_features = self_attn_layer(
                queries=vision_features,
                keys=vision_features,
                values=vision_features,
                boxes=boxes,
                padding_mask=vision_padding_mask
            )
            # then pass to the guided-attention layer
            vision_features = guided_attn_layer(
                queries=vision_features,
                keys=language_features,
                values=language_features,
                boxes=boxes,
                padding_mask=language_padding_mask
            )

        return vision_features

@META_ENCODER.register()
class CoAttentionEncoder(nn.Module):
    '''
        This module is designed inspired from ViLBERT ().
    '''
    def __init__(self, config):
        super(CoAttentionEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        # cross-attention layers
        self.vision_language_attn_layers = nn.ModuleDict([EncoderLayer(config.VISION_LANGUAGE_ATTENTION) for _ in range(config.LAYERS)])
        self.language_vision_attn_layers = nn.ModuleDict([EncoderLayer(config.LANGUAGE_VISION_ATTENTION) for _ in range(config.LAYERS)])

        # self-attention layers
        self.vision_self_attn_layers = nn.ModuleDict([EncoderLayer(config.VISION_SELF_ATTENTION) for _ in range(config.LAYERS)])
        self.language_self_attn_layers = nn.ModuleDict([EncoderLayer(config.LANGUAGE_SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, input_features):
        vision_features = input_features.vision_features
        boxes = input_features.boxes
        vision_padding_mask = input_features.vision_padding_mask

        language_features = input_features.language_padding_mask
        language_padding_mask = input_features.language_padding_mask

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
                boxes=boxes,
                attention_mask=vision_padding_mask
            )
            language_features = language_vision_attn_layer(
                queries=language_features,
                keys=vision_features,
                values=vision_features,
                attention_mask=language_padding_mask
            )
            # performing self-attention
            vision_features = vision_self_attn_layer(
                queries=vision_features,
                keys=vision_features,
                values=vision_features,
                boxes=boxes,
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
    def __init__(self, config):
        super(CoAttentionEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([CrossModalityEncoderLayer(config) for _ in range(config.LAYERS)])

    def forward(self, input_features: Instances):
        vision_features = input_features.vision_features
        boxes = input_features.boxes
        vision_padding_mask = input_features.vision_padding_mask

        language_features = input_features.language_padding_mask
        language_padding_mask = input_features.language_padding_mask

        for layer in self.layers:
            vision_features, language_features = layer(
                vision_features=vision_features,
                vision_padding_mask=vision_padding_mask,
                boxes=boxes,
                language_features=language_features,
                language_padding_mask=language_padding_mask
            )
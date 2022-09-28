import torch
from torch import nn

from .base_transformer import BaseTransformer
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class M4C(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)

        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        self.self_encoder = build_encoder(config.SELF_ENCODER)        

    def forward(self, input_features: Instances):
        pass

    def encoder_forward(self, input_features: Instances):
        pass
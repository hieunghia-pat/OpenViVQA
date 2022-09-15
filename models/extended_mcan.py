import torch

from .base_transformer import BaseTransformer
from data_utils.vocab import Vocab
from utils.instances import Instances
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class ExtendedMCAN(BaseTransformer):
    def __init__(self, config, vocab: Vocab):
        super(ExtendedMCAN, self).__init__(vocab)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.encoder = build_encoder(config.ENCODER)
        self.decoder = build_decoder(config.DECODER, vocab=vocab)

    def forward(self, input_features: Instances):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        encoder_features = self.encoder(Instances(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        ))

        answer_tokens = input_features.answer_tokens
        output = self.decoder(Instances(
            answer_tokens=answer_tokens,
            encoder_features=encoder_features,
            encoder_attention_mask=vision_padding_mask
        ))

        return output

    def encoder_forward(self, input_features: Instances):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, text_padding_mask, _ = self.text_embedding(question_tokens)

        encoder_features = self.encoder(Instances(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        ))

        return encoder_features, vision_padding_mask
    
    def step(self, t, prev_output):
        bs = self.encoder_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.encoder_features.device)
        else:
            it = prev_output

        output = self.decoder(Instances(
            answer_tokens=it,
            enc_features=self.encoder_features,
            encoder_attention_mask=self.encoder_padding_mask
        ))

        return output
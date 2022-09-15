from builders.encoder_builder import META_ENCODER
from .registry import Registry

META_DECODER = Registry("DECODER_LAYER")

def build_decoder(config, vocab):
    decoder = META_ENCODER.get(config.ARCHITECTURE)(config, vocab)

    return decoder
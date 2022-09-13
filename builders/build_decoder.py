from builders.build_encoder import META_ENCODER
from .registry import Registry

META_DECODER = Registry("DECODER_LAYER")

def build_decoder(config):
    decoder = META_ENCODER.get(config.MODEL.DECODER.ARCHITECTURE)(config)

    return decoder
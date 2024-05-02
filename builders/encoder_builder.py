from .registry import Registry

META_ENCODER = Registry("ENCODER_LAYER")

def build_encoder(config):
    encoder = META_ENCODER.get(config.architecture)(config)
    
    return encoder
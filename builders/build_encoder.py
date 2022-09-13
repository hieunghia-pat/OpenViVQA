from .registry import Registry

META_ENCODER = Registry("ENCODER_LAYER")

def build_encoder(config):
    encoder = Registry.get(config.MODEL.ENCODER.ARCHITECTURE)(config)
    
    return encoder
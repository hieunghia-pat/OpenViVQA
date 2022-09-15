from .registry import Registry

META_VISION_EMBEDDING = Registry("META_VISION_EMBEDDING")

def build_vision_embedding(config):
    vision_embedding = META_VISION_EMBEDDING.get(config.ARCHITECTURE)(config)

    return vision_embedding
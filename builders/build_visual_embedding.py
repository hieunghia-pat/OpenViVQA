from .registry import Registry

META_VISUAL_EMBEDDING = Registry("VISUAL_EMBEDDING")

def build_embedding(config):
    embedding = META_VISUAL_EMBEDDING.get(config.MODEL.EMBEDDING.VISUAL_EMBEDDING)(config)

    return embedding
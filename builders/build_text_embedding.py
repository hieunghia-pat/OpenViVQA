from .registry import Registry

META_TEXT_EMBEDDING = Registry("TEXT_EMBEDDING")

def build_embedding(config):
    embedding = META_TEXT_EMBEDDING.get(config.MODEL.EMBEDDING.TEXT_EMBEDDING)(config)

    return embedding
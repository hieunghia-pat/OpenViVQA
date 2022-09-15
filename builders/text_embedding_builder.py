from .registry import Registry

META_TEXT_EMBEDDING = Registry("TEXT_EMBEDDING")

def build_text_embedding(config, vocab):
    embedding = META_TEXT_EMBEDDING.get(config.ARCHITECTURE)(config, vocab)

    return embedding
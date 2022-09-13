from .registry import Registry

META_WORD_EMBEDDING = Registry("WORD_EMBEDDING")

def build_word_embedding(config):
    word_embedding = META_WORD_EMBEDDING.get(config.DATASET.WORD_EMBEDDING)(config)

    return word_embedding

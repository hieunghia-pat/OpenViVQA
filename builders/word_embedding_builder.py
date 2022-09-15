from .registry import Registry

META_WORD_EMBEDDING = Registry("WORD_EMBEDDING")

def build_word_embedding(config):
    word_embedding_names = config.VOCAB.WORD_EMBEDDING
    if isinstance(word_embedding_names, list):
        word_embedding = []
        for word_embedding_name in word_embedding_names:
            word_embedding.append(META_WORD_EMBEDDING.get(word_embedding_name)(cache=config.VOCAB.WORD_EMBEDDING_CACHE))
    else:
        word_embedding = META_WORD_EMBEDDING.get(config.VOCAB.WORD_EMBEDDING)(cache=config.VOCAB.WORD_EMBEDDING_CACHE)

    return word_embedding

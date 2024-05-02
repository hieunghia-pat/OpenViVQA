from .registry import Registry

META_WORD_EMBEDDING = Registry("WORD_EMBEDDING")

def build_word_embedding(config):
    word_embedding_names = config.word_embedding
    if isinstance(word_embedding_names, list):
        word_embedding = []
        for word_embedding_name in word_embedding_names:
            word_embedding.append(META_WORD_EMBEDDING.get(word_embedding_name)(cache=config.word_embedding_cache))
    else:
        word_embedding = META_WORD_EMBEDDING.get(config.word_embedding)(cache=config.word_embedding_cache)

    return word_embedding

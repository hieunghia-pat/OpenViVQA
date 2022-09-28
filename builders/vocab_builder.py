from .registry import Registry

META_VOCAB = Registry("VOCAB")

def build_vocab(config):
    vocab = META_VOCAB.get(config.VOCAB.TYPE)(config)

    return vocab
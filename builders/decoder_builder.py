from .registry import Registry

META_DECODER = Registry("DECODER_LAYER")

def build_decoder(config, vocab):
    decoder = META_DECODER.get(config.architecture)(config, vocab)

    return decoder
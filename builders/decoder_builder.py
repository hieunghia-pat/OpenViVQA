from .registry import Registry

META_DECODER = Registry("DECODER_LAYER")

def build_decoder(config, vocab):
    decoder = META_DECODER.get(config.ARCHITECTURE)(config, vocab)

    return decoder
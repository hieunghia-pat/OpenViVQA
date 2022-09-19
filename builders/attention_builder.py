from .registry import Registry

META_ATTENTION = Registry("META_ATTENTION")

def build_attention(config):
    attention_module = META_ATTENTION.get(config.ARCHITECTURE)(config)

    return attention_module
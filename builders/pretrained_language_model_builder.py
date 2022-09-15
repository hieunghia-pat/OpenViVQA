from .registry import Registry

META_PRETRAINED_LANGUAGE_MODEL = Registry("PRETRAINED_LANGUAGE_MODEL")

def build_pretrained_language_model(config):
    pretrained_language_model = META_PRETRAINED_LANGUAGE_MODEL.get(config.ARCHITECTTURE)(config)

    return pretrained_language_model
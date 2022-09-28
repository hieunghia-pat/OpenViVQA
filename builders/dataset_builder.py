from .registry import Registry

META_DATASET = Registry("DATASET")

def build_dataset(json_path, vocab, config):
    dataset = META_DATASET.get(config.TYPE)(json_path, vocab, config)

    return dataset
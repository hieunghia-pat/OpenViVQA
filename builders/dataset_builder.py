from .registry import Registry

META_DATASET = Registry("DATASET")

def build_dataset(json_path, vocab, config):
    if json_path is None:
        return None
    
    dataset = META_DATASET.get(config.type)(json_path, vocab, config)

    return dataset
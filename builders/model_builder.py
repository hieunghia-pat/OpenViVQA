import torch
from .registry import Registry

META_ARCHITECTURE = Registry(name="ARCHITECTURE")

def build_model(config, vocab):
    model = META_ARCHITECTURE.get(config.architecture)(config, vocab)
    model = model.to(torch.device(config.device))
    
    return model
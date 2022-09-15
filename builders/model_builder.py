import torch
from .registry import Registry

META_ARCHITECTURE = Registry(name="ARCHITECTURE")

def build_model(config):
    model = META_ARCHITECTURE.get(config.ARCHITECTURE)(config)
    model = model.to(torch.device(config.DEVICE))
    
    return model
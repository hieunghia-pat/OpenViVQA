import torch
from .registry import Registry

META_ARCHITECTURE = Registry(name="ARCHITECTURE")

def build_model(config):
    model = META_ARCHITECTURE.get(config.MODEL.ARCHITECTURE)(config)
    model = model.to(torch.device(config.MODEL.DEVICE))
    
    return model
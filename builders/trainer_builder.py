from .registry import Registry

META_TRAINER = Registry("TRAINER")

def build_trainer(config):
    trainer = META_TRAINER.get(config.TASK)(config)

    return trainer
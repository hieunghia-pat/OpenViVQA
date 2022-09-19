from .registry import Registry

META_TASK = Registry("TASK")

def build_task(config):
    task = META_TASK.get(config.TASK)(config)

    return task
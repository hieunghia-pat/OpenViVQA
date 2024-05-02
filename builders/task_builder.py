from .registry import Registry
from tasks.base_task import BaseTask

META_TASK = Registry("TASK")

def build_task(config) -> BaseTask:
    task = META_TASK.get(config.task)(config)

    return task
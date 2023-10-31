from utils.logging_utils import setup_logger
from .open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
from evaluation import Cider

logger = setup_logger()

@META_TASK.register()
class TrainingSAAATask(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        self.train_cider = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset.answers)})

    def lambda_lr(self, step):
        return self.learning_rate
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.logging_utils import setup_logger
from utils.instance import Instance
from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from data_utils.datasets.vitextcap_dataset import ViTextCapsDataset
import evaluation
from evaluation import Cider

import os
import numpy as np
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

logger = setup_logger()

@META_TASK.register()
class TrainingCaption(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def load_datasets(self, config):
        """Load datasets for the task."""
        train_dataset = ViTextCapsDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        
        dev_dataset = ViTextCapsDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = ViTextCapsDataset(config.JSON_PATH.TEST, self.vocab, config)
        

        self.train_dataset, self.dev_dataset, self.test_dataset = (
            train_dataset,
            dev_dataset,
            test_dataset,
        )

    def create_feature_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )

        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )

    def create_dataloaders(self, config):
        self.create_feature_dataloaders(config)


    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        print(self.train_dataset)
        self.train_cider = Cider(
            {
                f"{idx}": sample.answer
                for idx, sample in enumerate(self.train_dataset)
            }
        )



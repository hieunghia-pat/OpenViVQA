import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from builders.vocab_builder import build_vocab

from utils.logging_utils import Logger
from builders.model_builder import build_model

import os
import numpy as np
import pickle
import random

class BaseTask:
    def __init__(self, config):
        self.checkpoint_path = os.path.join(config.training.checkpoint_path, config.model.name)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        log_file = os.path.join(config.training.checkpoint_path, 
                       config.model.name,
                       f"task_{config.model.name}.log")
        self.logger = Logger(log_file)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            self.logger.info("Creating vocab")
            self.vocab = self.load_vocab(config.dataset.vocab)
            self.logger.info("Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        else:
            self.logger.info("Loading vocab from %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            self.vocab = pickle.load(open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb"))

        self.logger.info("Loading data")
        self.load_datasets(config.dataset)
        self.create_dataloaders(config.dataset)

        self.logger.info("Building model")
        self.model = build_model(config.model, self.vocab)
        self.config = config
        self.device = config.model.device

        self.logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)
        self.optim = Adam(self.model.parameters(), lr=config.training.learning_rate, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_token_idx)

    def configuring_hyperparameters(self, config):
        raise NotImplementedError

    def load_vocab(self, config):
        vocab = build_vocab(config)

        return vocab
    
    def load_datasets(self, config):
        raise NotImplementedError

    def create_dataloaders(self, config):
        raise NotImplementedError

    def evaluate_loss(self, dataloader: DataLoader):
        raise NotImplementedError

    def evaluate_metrics(self, dataloader: DataLoader):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.config.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        self.logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        self.logger.info("Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def start(self):
        raise NotImplementedError

    def get_predictions(self, dataset, get_scores=True):
        raise NotImplementedError

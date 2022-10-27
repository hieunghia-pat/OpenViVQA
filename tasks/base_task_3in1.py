import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from builders.vocab_builder import build_vocab

from utils.logging_utils import setup_logger
from builders.model_builder import build_model

import os
import numpy as np
import pickle
import random

logger = setup_logger()

class BaseTask:
    def __init__(self, config):
    
        self.checkpoint_path_en= os.path.join(config.TRAINING.CHECKPOINT_PATH_EN, config.MODEL.NAME)
        self.checkpoint_path_vi= os.path.join(config.TRAINING.CHECKPOINT_PATH_VI, config.MODEL.NAME)
        self.checkpoint_path_ja= os.path.join(config.TRAINING.CHECKPOINT_PATH_JA, config.MODEL.NAME)
        # if not os.path.isdir(self.checkpoint_path_en):
        #     logger.info("Creating checkpoint path")
        #     os.makedirs(self.checkpoint_path_en)

        # if not os.path.isfile(os.path.join(self.checkpoint_path_en, "vocab.bin")):
        #     logger.info("Creating vocab")
        #     self.vocab = self.load_vocab(config.DATASET)
        #     logger.info("Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
        #     pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        # else:
#ENGLISH
        if self.checkpoint_path_en is not None:
          logger.info("Loading vocab from en %s" % os.path.join(self.checkpoint_path_en, "vocab_en.bin"))
          self.vocab_en = pickle.load(open(os.path.join(self.checkpoint_path_en, "vocab_en.bin"), "rb"))
    
          logger.info("Loading data")
          self.load_datasets_en=self.load_datasets_en(config.DATASET)
          self.create_dataloaders_en=self.create_dataloaders_en(config)

          logger.info("Building model")
          self.model_en = build_model(config.MODEL, self.vocab_en)
          self.config_en = config
          self.device_en = torch.device(config.MODEL.DEVICE)

          logger.info("Defining optimizer and objective function")
          self.configuring_hyperparameters_en(config)
          self.optim_en = Adam(self.model_en.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))
          self.scheduler_en = LambdaLR(self.optim_en, self.lambda_lr_en)
          self.loss_fn_en = NLLLoss(ignore_index=self.vocab_en.padding_idx)
#VIETNAMESE
        if self.checkpoint_path_vi is not None:
          logger.info("Loading vocab from en %s" % os.path.join(self.checkpoint_path_vi, "vocab_vi.bin"))
          self.vocab_vi = pickle.load(open(os.path.join(self.checkpoint_path_vi, "vocab_vi.bin"), "rb"))
    
          logger.info("Loading data")
          self.load_datasets_vi=self.load_datasets_vi(config.DATASET)
          self.create_dataloaders_vi=self.create_dataloaders_vi(config)

          logger.info("Building model")
          self.model_vi = build_model(config.MODEL, self.vocab_vi)
          self.config_vi= config
          self.device_vi= torch.device(config.MODEL.DEVICE)

          logger.info("Defining optimizer and objective function")
          self.configuring_hyperparameters_vi(config)
          self.optim_vi = Adam(self.model_vi.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))
          self.scheduler_vi = LambdaLR(self.optim_vi, self.lambda_lr_vi)
          self.loss_fn_vi= NLLLoss(ignore_index=self.vocab_vi.padding_idx)
  #JANPANESE
        if self.checkpoint_path_ja is not None:
          logger.info("Loading vocab from en %s" % os.path.join(self.checkpoint_path_ja, "vocab_ja.bin"))
          self.vocab_ja = pickle.load(open(os.path.join(self.checkpoint_path_ja, "vocab_ja.bin"), "rb"))
    
          logger.info("Loading data")
          self.load_datasets_ja=self.load_datasets_ja(config.DATASET)
          self.create_dataloaders_ja=self.create_dataloaders_ja(config)

          logger.info("Building model")
          self.model_ja= build_model(config.MODEL, self.vocab_ja)
          self.config_ja= config
          self.device_ja= torch.device(config.MODEL.DEVICE)

          logger.info("Defining optimizer and objective function")
          self.configuring_hyperparameters_ja(config)
          self.optim_ja= Adam(self.model_ja.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))
          self.scheduler_ja = LambdaLR(self.optim_ja, self.lambda_lr_ja)
          self.loss_fn_ja = NLLLoss(ignore_index=self.vocab_ja.padding_idx)
#ENGLISH
    def configuring_hyperparameters_en(self, config):
        raise NotImplementedError
#VIETNAMESE
    def configuring_hyperparameters_vi(self, config):
        raise NotImplementedError
#JAPANESE
    def configuring_hyperparameters_ja(self, config):
          raise NotImplementedError

    def load_vocab(self, config):
        vocab = build_vocab(config)

        return vocab
    
    def load_datasets_en(self, config):
        raise NotImplementedError
    def load_datasets_vi(self, config):
        raise NotImplementedError
    def load_datasets_ja(self, config):
        raise NotImplementedError


    def create_dataloaders_en(self, config):
        raise NotImplementedError
    def create_dataloaders_vi(self, config):
        raise NotImplementedError
    def create_dataloaders_ja(self, config):
        raise NotImplementedError
#ENGLISH
    def evaluate_loss_en(self, dataloader: DataLoader):
        raise NotImplementedError
    def evaluate_metrics_en(self, dataloader: DataLoader):
        raise NotImplementedError
#VIETNAMESE
    def evaluate_loss_vi(self, dataloader: DataLoader):
        raise NotImplementedError
    def evaluate_metrics_vi(self, dataloader: DataLoader):
        raise NotImplementedError
#JAPANESE
    def evaluate_loss_ja(self, dataloader: DataLoader):
        raise NotImplementedError
    def evaluate_metrics_ja(self, dataloader: DataLoader):
        raise NotImplementedError

    def train_en(self):
        raise NotImplementedError
    def train_vi(self):
        raise NotImplementedError
    def train_ja(self):
      raise NotImplementedError
#ENGLISH
    def lambda_lr_en(self, step):
        warm_up = self.warmup_en
        step += 1
        return (self.model_en.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)
#VIETNAMESE
    def lambda_lr_vi(self, step):
        warm_up = self.warmup_vi
        step += 1
        return (self.model_vi.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)
#JAPANESE
    def lambda_lr_ja(self, step):
        warm_up = self.warmup_ja
        step += 1
        return (self.model_ja.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)
#ENGLISH
    def load_checkpoint_en(self, fname_en) -> dict:
        if not os.path.exists(fname_en):
            return None

        logger.info("Loading checkpoint from %s", fname_en)

        checkpoint = torch.load(fname_en)

        torch.set_rng_state(checkpoint['torch_rng_state_en'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state_en'])
        np.random.set_state(checkpoint['numpy_rng_state_en'])
        random.setstate(checkpoint['random_rng_state_en'])

        self.model_en.load_state_dict(checkpoint['state_dict_en'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch_en'])

        return checkpoint
    def save_checkpoint_en(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state_en': torch.get_rng_state(),
            'cuda_rng_state_en': torch.cuda.get_rng_state(),
            'numpy_rng_state_en': np.random.get_state(),
            'random_rng_state_en': random.getstate(),
            'epoch_en': self.epoch_en,
            'state_dict_en': self.model_en.state_dict(),
            'optimizer_en': self.optim_en.state_dict(),
            'scheduler_en': self.scheduler_en.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path_en, "last_model.pth"))
        torch.save(dict_for_saving, os.path.join('/content/drive/MyDrive/', "last_model.pth"))
#VIETNAMESE
    def load_checkpoint_vi(self, fname_vi) -> dict:
        if not os.path.exists(fname_vi):
            return None

        logger.info("Loading checkpoint from %s", fname_vi)

        checkpoint = torch.load(fname_vi)

        torch.set_rng_state(checkpoint['torch_rng_state_vi'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state_vi'])
        np.random.set_state(checkpoint['numpy_rng_state_vi'])
        random.setstate(checkpoint['random_rng_state_vi'])

        self.model_vi.load_state_dict(checkpoint['state_dict_vi'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch_vi'])

        return checkpoint

    def save_checkpoint_vi(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state_vi': torch.get_rng_state(),
            'cuda_rng_state_vi': torch.cuda.get_rng_state(),
            'numpy_rng_state_vi': np.random.get_state(),
            'random_rng_state_vi': random.getstate(),
            'epoch_vi': self.epoch_vi,
            'state_dict_vi': self.model_vi.state_dict(),
            'optimizer_vi': self.optim_vi.state_dict(),
            'scheduler_vi': self.scheduler_vi.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path_vi, "last_model.pth"))
        torch.save(dict_for_saving, os.path.join('/content/drive/MyDrive/', "last_model.pth"))
#JAPANESE
    def load_checkpoint_ja(self, fname_ja) -> dict:
        if not os.path.exists(fname_ja):
            return None

        logger.info("Loading checkpoint from %s", fname_ja)

        checkpoint = torch.load(fname_ja)

        torch.set_rng_state(checkpoint['torch_rng_state_ja'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state_ja'])
        np.random.set_state(checkpoint['numpy_rng_state_ja'])
        random.setstate(checkpoint['random_rng_state_ja'])

        self.model_ja.load_state_dict(checkpoint['state_dict_ja'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch_ja'])

        return checkpoint       
    
    def save_checkpoint_ja(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state_ja': torch.get_rng_state(),
            'cuda_rng_state_ja': torch.cuda.get_rng_state(),
            'numpy_rng_state_ja': np.random.get_state(),
            'random_rng_state_ja': random.getstate(),
            'epoch_ja': self.epoch_ja,
            'state_dict_ja': self.model_ja.state_dict(),
            'optimizer_ja': self.optim_ja.state_dict(),
            'scheduler_ja': self.scheduler_ja.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path_ja, "last_model.pth"))
        torch.save(dict_for_saving, os.path.join('/content/drive/MyDrive/', "last_model.pth"))



    def start_en(self):
        raise NotImplementedError

    def start_vi(self):
        raise NotImplementedError
  
    def start_ja(self):
        raise NotImplementedError

    def get_predictions_en(self, dataset, get_scores=True):
        raise NotImplementedError

    def get_predictions_vi(self, dataset, get_scores=True):
        raise NotImplementedError
  
    def get_predictions_ja(self, dataset, get_scores=True):
        raise NotImplementedError

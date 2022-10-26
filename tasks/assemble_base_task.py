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

class AssembleBaseTask:
    def __init__(self, config):

        self.checkpoint_path = [os.path.join(config.PRETRAINED_MODELS.E_MODEL_PATH, config.MODEL.NAME),
                                os.path.join(config.PRETRAINED_MODELS.V_MODEL_PATH, config.MODEL.NAME),
                                os.path.join(config.PRETRAINED_MODELS.J_MODEL_PATH, config.MODEL.NAME)]
        self.vocab = []
        for ckpt_path in self.checkpoint_path:
            if not os.path.isdir(ckpt_path):
                raise Exception(f"{ckpt_path} does not exist")
            if not os.path.isfile(os.path.join(ckpt_path, "vocab.bin")):
                raise Exception(f"vocab file does not exist")
            else:
                logger.info("Loading vocab from %s" % os.path.join(ckpt_path, "vocab.bin"))
                self.vocab.append(pickle.load(open(os.path.join(ckpt_path, "vocab.bin"), "rb")))

        logger.info("Loading data")
        self.load_datasets(config.DATASET)
        self.create_dataloaders(config)

        logger.info("Building model")
        self.model = [build_model(config.MODEL, self.vocab[0]), 
                    build_model(config.MODEL, self.vocab[1]),
                    build_model(config.MODEL, self.vocab[2])]
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)
        # self.optim = [Adam(self.model[0].parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98)),
        #             Adam(self.model[1].parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98)),
        #             Adam(self.model[2].parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))]
        
        
        # self.scheduler = [LambdaLR(self.optim[0], self.e_lambda_lr),
        #                 LambdaLR(self.optim[1], self.v_lambda_lr),
        #                 LambdaLR(self.optim[2], self.j_lambda_lr)]

        # self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)

    def configuring_hyperparameters(self, config):
        raise NotImplementedError

    def load_vocab(self, config):
        vocab = build_vocab(config)

        return vocab
    
    def load_datasets(self, config):
        raise NotImplementedError

    # def create_dataloaders(self, config):
    #     raise NotImplementedError

    # def evaluate_loss(self, dataloader: DataLoader):
    #     raise NotImplementedError

    # def evaluate_metrics(self, dataloader: DataLoader):
    #     raise NotImplementedError

    def train(self):
        raise NotImplementedError

    
    def load_checkpoint(self, fname, index) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model[index].load_state_dict(checkpoint['state_dict'], strict=False)

        logger.info("Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint

    
    def start(self):
        raise NotImplementedError

    def get_predictions(self, dataset, get_scores=True):
        raise NotImplementedError

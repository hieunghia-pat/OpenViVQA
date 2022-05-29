import torch
import os
import dill as pickle
import numpy as np
import random
import json

from trainers.answering_model_trainer import Trainer
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.utils import collate_fn
from configs.config import get_default_config

from models.modules.transformer import FusionTransformer

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--config-file", type=str, default="configs/base.yaml")
parser.add_argument("--start-from", default=None, type=str)

args = parser.parse_args()

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

config = get_default_config()
config.merge_from_file(args.config_file)

if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Checking checkpoint directory...")
# creating checkpoint directory
if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl")):
    print("Creating vocab...")
    vocab = Vocab(
                    json_dirs = [
                        config.path.train_json_path,
                        config.path.dev_json_path,
                        config.path.test_json_path
                    ],
                    min_freq = config.dataset.min_freq,
                    pretrained_language_model_name = config.model.pretrained_language_model_name,
                    tokenizer_name = config.dataset.tokenizer
                )
    pickle.dump(vocab, open(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl"), "wb"))
else:
    print("Loading the vocab...")
    vocab = pickle.load(open(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl"), "rb"))

print("Creating dataset...")
# creating iterable dataset
train_dataset = FeatureDataset(
                                json_path = config.path.train_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = vocab,
                                tokenizer_name = config.dataset.tokenizer
                            )

val_dataset = FeatureDataset(
                                json_path = config.path.dev_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = vocab,
                                tokenizer_name = config.dataset.tokenizer
                            )

test_dataset = FeatureDataset(
                                json_path = config.path.test_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = vocab,
                                tokenizer_name = config.dataset.tokenizer
                            )

# creating dictionary dataset
train_dict_dataset = DictionaryDataset(
                                        json_path = config.path.train_json_path,
                                        image_features_path=config.path.image_features_path,
                                        vocab = vocab,
                                        tokenizer_name = config.dataset.tokenizer
                                    )

val_dict_dataset = DictionaryDataset(
                                        json_path = config.path.dev_json_path,
                                        image_features_path=config.path.image_features_path,
                                        vocab = vocab,
                                        tokenizer_name = config.dataset.tokenizer
                                    )

test_dict_dataset = DictionaryDataset(
                                        json_path = config.path.test_json_path,
                                        image_features_path=config.path.image_features_path,
                                        vocab = vocab,
                                        tokenizer_name = config.dataset.tokenizer
                                    )

print("Initializing model...")
model = FusionTransformer(vocab, config).to(device)

print("Defining the trainers...")
# Define Trainer
trainer = Trainer(model=model, train_datasets=(train_dataset, train_dict_dataset), val_datasets=(val_dataset, val_dict_dataset),
                    test_datasets=(test_dataset, test_dict_dataset), vocab=vocab, collate_fn=collate_fn, config=config)

# Training
if args.start_from:
    trainer.train(os.path.join(config.training.checkpoint_path, config.model.name, args.start_from))
else:
    trainer.train()

results = trainer.get_predictions(test_dict_dataset,
                                    checkpoint_filename=os.path.join(config.training.checkpoint_path, config.model.name, "best_model.pth"),
                                    get_scores=config.training.get_scores)
json.dump(results, open(os.path.join(config.training.checkpoint_path, config.model.name, "test_results.json"), "w+"), ensure_ascii=False)
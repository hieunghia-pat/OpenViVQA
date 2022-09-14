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

from models.transformers import FusionTransformer

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

config = get_default_config()

if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

device = "cuda" if torch.cuda.is_available() else "cpu"

# creating checkpoint directory
if not os.path.isdir(os.path.join(config.training.checkpoint_path, config.model.name)):
    os.makedirs(os.path.join(config.training.checkpoint_path, config.model.name))

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl")):
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
    vocab = pickle.load(open(os.path.join(config.training.checkpoint_path, config.model.name, "vocab.pkl"), "rb"))

# creating iterable dataset
train_dataset = FeatureDataset(
                                json_path = config.path.train_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = None,
                                tokenizer_name = config.dataset.tokenizer
                            )

val_dataset = FeatureDataset(
                                json_path = config.path.test_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = None,
                                tokenizer_name = config.dataset.tokenizer
                            )

test_dataset = FeatureDataset(
                                json_path = config.path.test_json_path,
                                image_features_path = config.path.image_features_path,
                                vocab = None,
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
                                        json_path = config.path.val_json_path,
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

model = FusionTransformer(vocab, config).to(device)

# Define Trainer
trainer = Trainer(model=model, train_datasets=(train_dataset, train_dict_dataset), val_datasets=(val_dataset, val_dict_dataset),
                    test_datasets=(test_dataset, test_dict_dataset), vocab=vocab, collate_fn=collate_fn)

# Training
if config.training.start_from:
    trainer.train(os.path.join(config.training.checkpoint_path, config.model.name, config.training.start_from))
else:
    trainer.train()

results = trainer.get_predictions(test_dict_dataset,
                                    checkpoint_filename=os.path.join(config.training.checkpoint_path, config.model.name, config.training.start_from),
                                    get_scores=config.training.get_scores)
json.dump(results, open(os.path.join(config.checkpoint_path, config.model_name, "test_results.json"), "w+"), ensure_ascii=False)
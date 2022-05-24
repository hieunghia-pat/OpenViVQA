import torch
import os
import dill as pickle
import numpy as np
import random
import config

from training_utils.language_model_trainer import Trainer
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset
from data_utils.utils import collate_fn

random.seed(13)
torch.manual_seed(13)
np.random.seed(13)

# creating checkpoint directory
if not os.path.isdir(os.path.join(config.checkpoint_path, config.model_name)):
    os.makedirs(os.path.join(config.checkpoint_path, config.model_name))

if not os.path.isdir(os.path.join(config.checkpoint_path, config.model_name)):
    os.makedirs(os.path.join(config.checkpoint_path, config.model_name))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating vocabulary and dataset
if not os.path.isfile(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl")):
    vocab = Vocab([config.train_json_path, config.val_json_path], tokenizer_name=config.tokenizer, 
                    pretrained_language_model_name=config.pretrained_language_model_name)
    pickle.dump(vocab, open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "wb"))
else:
    vocab = pickle.load(open(os.path.join(config.checkpoint_path, config.model_name, "vocab.pkl"), "rb"))

# creating iterable dataset
train_dataset = FeatureDataset(config.train_json_path, \
                                config.region_features_path, \
                                config.grid_features_path, \
                                config.mask_features_path, \
                                config.guided_load_feature,
                                49,
                                vocab,
                                config.idx_by_filename) # for training with cross-entropy loss

val_dataset = FeatureDataset(config.val_json_path, \
                                config.region_features_path, \
                                config.grid_features_path, \
                                config.mask_features_path, \
                                config.guided_load_feature,
                                49,
                                vocab,
                                config.idx_by_filename)  # for calculating evaluation loss
                                
if config.public_test_json_path is not None:
    public_test_dataset = FeatureDataset(config.public_test_json_path, config.feature_path, vocab=vocab)
else:
    public_test_dataset = None

model = config.pretrained_language_model(config.pretrained_language_model_name, padding_idx=vocab.padding_idx, bert_hidden_size=config.language_model_hidden_size, 
                        vocab_size=len(vocab), d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, h=config.nhead, d_ff=config.d_ff,
                        max_len=vocab.max_caption_length, dropout=config.dropout).to(device)

trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                    test_dataset=public_test_dataset, vocab=vocab, collate_fn=collate_fn)

if config.start_from:
    trainer.train(os.path.join(config.checkpoint_path, config.model_name, config.start_from))
else:
    trainer.train()
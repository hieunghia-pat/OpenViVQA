from configs.utils import get_config
from transformers.models.bert.modeling_bert import BertConfig
import torch
from torch.utils.data import DataLoader
from data_utils.utils import collate_fn
from models.mma_sr import MMA_SR_Model
import pickle
import os
import numpy as np
import torch
from typing import Dict, Any
from dummy_dataset import ViTextCapsDataset

mmt_config = BertConfig(hidden_size=768,
                        num_hidden_layers=3)

vocab = pickle.load(open(r"data\\vocab.bin", "rb"))
config_file = r'OpenViVQA/configs/mma_sr.yaml'
config = get_config(config_file)

dataset  = ViTextCapsDataset('data\\dev.json',
                              vocab=vocab,
                              config=config)

train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn
        )
item = next(iter(train_dataloader))
print(item)
model = MMA_SR_Model(config.MODEL, vocab)
model.build()
model.eval()
with torch.inference_mode():
    output = model.forward(item)
print(output)


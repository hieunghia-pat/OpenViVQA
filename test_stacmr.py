from models.stacmr import VSRN
from data_utils.datasets.vitextcap_dataset import ViTextCapsDataset
import pickle
from configs.utils import get_config
from builders.vocab_builder import build_vocab
import torch
from torch.utils.data import DataLoader
from data_utils.utils import collate_fn

config = get_config('vitextcaps-captioning\configs\stacmr.yaml')
vocab = build_vocab(config.DATASET.VOCAB)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VSRN(config.MODEL, vocab)
model.to(device)
# print(model)

train_data = ViTextCapsDataset(json_path='D:\\Research\\OpenViVQA\data\\vitextcaps_dev.json',
                               vocab=vocab,
                               config=config.DATASET)
train_loader = DataLoader(train_data,
                          batch_size=2,
                          shuffle=True,
                          collate_fn=collate_fn)
sample = next(iter(train_loader))
model.eval()
with torch.no_grad():
    output = model(sample, mode='inference')
print(output['scores'].shape)
print(output['scores'])
print(output['scores'].argmax(dim=-1))
print(output['predicted_token'])

import torch

from transformers import T5Tokenizer

from builders.vocab_builder import META_VOCAB
from data_utils.vocabs.vocab import Vocab
from data_utils.utils import preprocess_sentence

from typing import List
import json
import re

@META_VOCAB.register()
class T5OcrVocab(Vocab):
    '''
        This class is designed especially for VQA with reading comprehension
    '''
    def __init__(self, config):

        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(config.PRETRAINED_NAME)
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "sep_token": "<sep>"
        })
        
        self.tokenizer = tokenizer.tokenize
        self.padding_token = tokenizer.pad_token
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.unk_token = tokenizer.unk_token
        self.sep_token = tokenizer.sep_token

        self.stoi = tokenizer.get_vocab()
        self.itos = {id: token for token, id in self.stoi.items()}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]
    
    def encode_token(self, tokens: List[str]) -> torch.Tensor:
        encoded_tokens = []
        for token in tokens:
            encoded_tokens.append(self.stoi[token])

        return torch.Tensor(encoded_tokens).long()

    def decode_answer(self, answer_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            answer_vecs: (bs, max_length)
        '''
        answers = []
        for vec in answer_vecs:
            answer = "".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            answer = re.sub("▁", " ", answer)
            if join_words:
                answers.append(answer.strip())
            else:
                answers.append(answer.strip().split())

        return answers

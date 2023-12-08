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
        self.tokenizer = tokenizer.tokenize

        self.padding_token = tokenizer.pad_token
        self.bos_token = "<extra_id_0>"
        self.eos_token = tokenizer.eos_token
        self.unk_token = tokenizer.unk_token

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])

        self.stoi = tokenizer.get_vocab()
        self.itos = {id: token for token, id in self.stoi.items()}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

    def make_vocab(self, json_dirs):
        self.max_question_length = 0
        self.max_answer_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                for answer in ann["answers"]:
                    question = preprocess_sentence(ann["question"], self.tokenizer)
                    answer = preprocess_sentence(answer, self.tokenizer)
                    if len(question) + 2 > self.max_question_length:
                            self.max_question_length = len(question) + 2
                    if len(answer) + 2 > self.max_answer_length:
                        self.max_answer_length = len(answer) + 2
    
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

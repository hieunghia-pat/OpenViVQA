from transformers import BertTokenizer

from data_utils.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB
from data_utils.vocabs.vocab import Vocab

import json
from typing import List

@META_VOCAB.register()
class BERTVocab(Vocab):
    """
        Defines a vocabulary object that will be used to numericalize a field.
    """
    def __init__(self, config):

        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.tokenizer = tokenizer.tokenize

        self.padding_token = tokenizer.pad_token
        self.bos_token = tokenizer.sep_token
        self.eos_token = tokenizer.sep_token
        self.unk_token = tokenizer.unk_token

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])

        self.itos = tokenizer.ids_to_tokens
        self.stoi = {token: id for id, token in self.itos.items()}

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

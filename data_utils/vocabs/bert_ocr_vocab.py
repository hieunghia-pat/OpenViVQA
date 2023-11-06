import torch

from transformers import BertTokenizer

from builders.vocab_builder import META_VOCAB
from data_utils.vocabs.ocr_vocab import OcrVocab
from data_utils.utils import preprocess_sentence

from typing import List, Dict
import numpy as np
import json
import re

@META_VOCAB.register()
class BertOcrVocab(OcrVocab):
    '''
        This class is designed especially for VQA with reading comprehension
    '''
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

    def match_text_to_indices(self, text: List[str], oov2inds: Dict[str, int]):
        '''
            Match an text to a list of sequences of indices
            each index corresponds to either a fixed vocabulary or an OOV token
            (in the index address space, the OOV tokens are after the fixed vocab)
        '''
        indices = []
        for word in text:
            matched_inds = []
            # match answer to fixed vocab
            matched_inds.append(self.stoi[word])
            # match answer word to OOV if available
            if word in oov2inds:
                matched_inds.extend(oov2inds[word])
            indices.append(matched_inds[np.random.choice(len(matched_inds))])

        return indices

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

    def decode_answer(self, answer_vecs: torch.Tensor, list_ocr_tokens: List[List[str]], join_words=True) -> List[str]:
        '''
            answer_vecs: (bs, max_length)
        '''
        ocr_token_of = [{len(self.stoi)+idx: token for idx, token in enumerate(ocr_tokens)} for ocr_tokens in list_ocr_tokens]
        answers = []
        for batch, vec in enumerate(answer_vecs):
            answer = []
            for idx in vec.tolist():
                if idx in ocr_token_of[batch]:
                    word = ocr_token_of[batch][idx]
                else:
                    word = self.itos[idx]
                if word == self.eos_token:
                    break
                if word not in self.specials:
                    answer.append(word)
            answer = " ".join(answer)
            answer = re.sub(r"\s+#+", "", answer)

            if join_words:
                answers.append(answer)
            else:
                answers.append(answer.strip().split())

        return answers

    def decode_answer_with_determination(self, answer_vecs: torch.Tensor, list_ocr_tokens: List[List[str]], join_words=True) -> List[str]:
        '''
            This module is designed to determine if a selected token is in fixed vocab or from ocr token
            params:
                - answer_vecs: (bs, max_length)
        '''
        ocr_token_of = [{len(self.stoi)+idx: token for idx, token in enumerate(ocr_tokens)} for ocr_tokens in list_ocr_tokens]
        answers = []
        in_fixed_vocab = []
        for batch, vec in enumerate(answer_vecs):
            answer = []
            in_fixed_vocab_per_batch = []
            for idx in vec.tolist():
                if idx in ocr_token_of[batch]:
                    word = ocr_token_of[batch][idx]
                    in_fixed_vocab_per_batch.append(False)
                else:
                    word = self.itos[idx]
                    in_fixed_vocab_per_batch.append(True)
                if word == self.eos_token:
                    break
                if word not in self.specials:
                    answer.append(word)
            answer = " ".join(answer)
            answer = re.sub(r"\s+#+", "", answer)

            if join_words:
                answers.append(answer)
            else:
                answers.append(answer.strip().split())
            in_fixed_vocab.append(in_fixed_vocab_per_batch)

        return answers, in_fixed_vocab

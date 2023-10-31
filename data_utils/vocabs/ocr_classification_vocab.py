import torch

from builders.vocab_builder import META_VOCAB
from .classification_vocab import ClassificationVocab

import numpy as np
from typing import List, Union, Dict
from copy import deepcopy
from collections import defaultdict

@META_VOCAB.register()
class OcrClassificationVocab(ClassificationVocab):
    def __init__(self, config):
        super().__init__(config)

        self.num_choices = self.total_answers + config.MAX_SCENE_TEXT

    def match_text_to_index(self, text: List[str], oov2inds: Dict[str, int]) -> int:
        text = " ".join(text)
        indices = [self.atoi[text]]
        if text in oov2inds:
            indices.append(oov2inds[text])

        index = indices[np.random.choice(1)]

        return index

    def encode_answer(self, answer: List[str], ocr_tokens: List[str]) -> torch.Tensor:
        ocr_tokens = {self.total_answers+idx: token for idx, token in enumerate(ocr_tokens)}
        ocr2inds = defaultdict(list)
        for idx, token in ocr_tokens.items():
            ocr2inds[token].append(idx)
        answer = self.match_text_to_index(answer, ocr2inds)
        assert isinstance(answer, int), f"answer must be the type of int, get {type(answer)}"

        vec = torch.tensor([answer]).long()

        return vec

    def decode_answer(self, answer_vecs: torch.Tensor, list_ocr_tokens: List[List[str]], join_word=True) -> Union[List[str], List[List[str]]]:
        ocr_token_of = [{self.total_answers+idx: token for idx, token in enumerate(ocr_tokens)} for ocr_tokens in list_ocr_tokens]
        answers = []
        answer_vecs = answer_vecs.squeeze(-1)
        list_answers = answer_vecs.tolist()
        for batch, answer_idx in enumerate(list_answers):
            batch_ocr_token_of = ocr_token_of[batch]
            itoa = deepcopy(self.itoa)
            itoa.update(batch_ocr_token_of)
            answers.append(itoa[answer_idx] if join_word else itoa[answer_idx].split())

        return answers

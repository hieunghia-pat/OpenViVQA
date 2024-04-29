import torch
from typing import *
from transformers import AutoTokenizer

from data_utils.vocabs import Vocab

class PretrainedVocab(Vocab):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

    @property
    def padding_token(self) -> str:
        return self.tokenizer.pad_token
    
    @property
    def padding_token_idx(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def bos_token(self) -> str:
        return self.tokenizer.bos_token
    
    @property
    def bos_token_idx(self) -> int:
        return self.tokenizer.bos_token_id
    
    @property
    def eos_token(self) -> str:
        return self.tokenizer.eos_token
    
    @property
    def eos_token_idx(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def unk_token(self) -> str:
        return self.tokenizer.unk_token
    
    @property
    def unk_token_idx(self) -> str:
        return self.tokenizer.unk_token_id

    def encode_question(self, question: List[str]) -> torch.Tensor:
        tokenized_questions = self.tokenizer(
            question,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )

        return tokenized_questions.input_ids, tokenized_questions.attention_mask

    def encode_answer(self, answer: List[str]) -> torch.Tensor:
        tokenized_answers = self.tokenizer(
            answer,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )

        return tokenized_answers.input_ids, tokenized_answers.attention_mask

    def decode_question(self, question_vecs: torch.Tensor, join_words=True) -> List[str]:
        questions = self.tokenizer.batch_decode(
            question_vecs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return questions

    def decode_answer(self, answer_vecs: torch.Tensor, join_words=True) -> List[str]:
        answers = self.tokenizer.batch_decode(
            answer_vecs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return answers

import torch
from typing import *
from transformers import AutoTokenizer

from data_utils.vocabs import Vocab
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class PretrainedVocab(Vocab):
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)

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

    def encode_question(self, question: str, context: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        tokenized_questions = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt"
        )

        # tensor of input_ids is (n, seq_len), so squeeze its first dimension for later convenience
        return tokenized_questions.input_ids.squeeze(0), tokenized_questions.attention_mask.squeeze(0)
    
    def encode_tokens(self, tokens: str):
        tokenized_tokens = self.tokenizer(tokens, 
                                        padding="max_length",
                                        max_length=20,
                                        truncation=True,
                                        return_tensors="pt",
                                        add_special_tokens=False).input_ids

        return tokenized_tokens

    def encode_answer(self, answer: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        tokenized_answers = self.tokenizer(
            answer,
            add_special_tokens=True,
            return_tensors="pt"
        )

        # tensor of input_ids is (n, seq_len), so squeeze its first dimension for later convenience
        return tokenized_answers.input_ids.squeeze(0), tokenized_answers.attention_mask.squeeze(0)

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

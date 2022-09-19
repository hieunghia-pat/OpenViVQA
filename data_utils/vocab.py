import torch

from data_utils.utils import preprocess_sentence, unk_init
from builders.word_embedding_builder import build_word_embedding

from transformers import AutoTokenizer

from collections import defaultdict, Counter
import json
from typing import List

class Vocab(object):
    """
        Defines a vocabulary object that will be used to numericalize a field.
    """
    def __init__(self, config):

        self.tokenizer = config.VOCAB.TOKENIZER

        if config.VOCAB.PRETRAINED_LANGUAGE_MODEL is not None: # use special tokens and vocab from pretrained language model
            token_encoder = AutoTokenizer.from_pretrained(config.VOCAB.PRETRAINED_LANGUAGE_MODEL)
            self.padding_token = token_encoder.pad_token
            self.bos_token = token_encoder.bos_token
            self.eos_token = token_encoder.eos_token
            self.unk_token = token_encoder.unk_token
        else: # use defined special tokens
            self.padding_token = config.VOCAB.PAD_TOKEN
            self.bos_token = config.VOCAB.BOS_TOKEN
            self.eos_token = config.VOCAB.EOS_TOKEN
            self.unk_token = config.VOCAB.UNK_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        self.itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            self.itos.append(word)

        self.stoi = defaultdict()
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        if config.VOCAB.USE_MAPPING:
            assert config.VOCAB.PRETRAINED_LANGUAGE_MODEL is not None, "Pretrained language model is required if using map for vocab"
            self.mapping = defaultdict()
            # map from original vocab to pretrained language models vocab
            self.mapping.update({ori_idx: self.token_encoder.convert_tokens_to_ids(token) for ori_idx, token in enumerate(self.itos)})
            # map special tokens
            self.mapping[self.padding_idx] = token_encoder.encoder[self.padding_token]
            self.mapping[self.bos_idx] = token_encoder.ecoder[self.bos_token]
            self.mapping[self.eos_idx] = token_encoder.encoder[self.eos_token]
            self.mapping[self.unk_idx] = token_encoder.encoder[self.unk_token]
        else:
            self.mapping = None

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        self.max_question_length = 0
        self.max_answer_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"], self.tokenizer)
                answer = preprocess_sentence(ann["answer"], self.tokenizer)
                self.freqs.update(question)
                self.freqs.update(answer)
                if len(question) + 2 > self.max_question_length:
                        self.max_question_length = len(question) + 2
                if len(answer) + 2 > self.max_answer_length:
                    self.max_answer_length = len(answer) + 2

    def encode_question(self, question: str) -> torch.Tensor:
        """ Turn a question into a vector of indices and a question length """
        vec = torch.ones(self.max_question_length).long() * self.padding_idx
        for i, token in enumerate([self.bos_token] + question + [self.eos_token]):
            vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx
        return vec

    def encode_answer(self, answer: str) -> torch.Tensor:
        """ Turn a answer into a vector of indices and a question length """
        vec = torch.ones(self.max_answer_length).long() * self.padding_idx
        for i, token in enumerate([self.bos_token] + answer + [self.eos_token]):
            vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx
        return vec

    def decode_question(self, question_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            question_vecs: (bs, max_length)
        '''
        questions = []
        for vec in question_vecs:
            question = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                questions.append(question)
            else:
                questions.append(question.strip().split())

        return questions

    def decode_answer(self, answer_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            answer_vecs: (bs, max_length)
        '''
        answers = []
        for vec in answer_vecs:
            answer = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                answers.append(answer)
            else:
                answers.append(answer.strip().split())

        return answers

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.word_embeddings != other.word_embeddings:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_word_embeddings(self, word_embeddings):
        if not isinstance(word_embeddings, list):
            word_embeddings = [word_embeddings]

        tot_dim = sum(embedding.dim for embedding in word_embeddings)
        self.word_embeddings = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in word_embeddings:
                end_dim = start_dim + v.dim
                self.word_embeddings[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, word_embeddings, dim):
        """
        Set the word_embeddings for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `word_embeddings` input argument.
            word_embeddings: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the word_embeddings.
        """
        self.word_embeddings = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            we_index = stoi.get(token, None)
            if we_index is not None:
                self.word_embeddings[i] = word_embeddings[we_index]
            else:
                self.word_embeddings[i] = unk_init(self.word_embeddings[i])

class ClassificationVocab(Vocab):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711

    def __init__(self, config):

        super(ClassificationVocab, self).__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        itoa = set()
        self.max_question_length = 0
        self.max_answer_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"], self.tokenizer)
                answer = ann["answer"]
                answer = "_".join(answer.split())
                self.freqs.update(question)
                itoa.add(answer)
                if len(question) + 2 > self.max_question_length:
                        self.max_question_length = len(question) + 2

        self.itoa = {ith: answer for ith, answer in enumerate(itoa)}
        self.atoi = defaultdict()
        self.atoi.update({answer: ith for ith, answer in self.itoa.items()})

    def encode_answer(self, answer: str) -> torch.Tensor:
        return torch.tensor([self.atoi[answer]], dtype=torch.long)

    def decode_answer(self, answer_vecs: torch.Tensor) -> List[str]:
        answers = []
        list_answers = answer_vecs.tolist()
        for answer_idx in list_answers:
            answers.append(self.itoa[answer_idx])

        return answers
import torch

from data_utils.vector import Vectors
from data_utils.vector import pretrained_aliases
from data_utils.utils import preprocess_sentence, unk_init

from transformers import AutoTokenizer

from collections import defaultdict, Counter
import six
import json
from typing import List, Union

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, json_dirs, max_size=None, min_freq=1, bos_token="<bos>", eos_token="<eos>", padding_token="<pad>", unk_token="<unk>",
                    pretrained_language_model_name=None, use_mapping=False, vectors=None, unk_init=unk_init, vectors_cache=None, 
                    tokenizer_name: Union[str, None]=None):

        self.tokenizer = tokenizer_name

        if pretrained_language_model_name is not None: # use special tokens and vocab from pretrained language model
            token_encoder = AutoTokenizer.from_pretrained(pretrained_language_model_name)
            self.padding_token = token_encoder.pad_token
            self.bos_token = token_encoder.bos_token
            self.eos_token = token_encoder.eos_token
            self.unk_token = token_encoder.unk_token
        else: # use defined special tokens
            self.padding_token = padding_token
            self.bos_token = bos_token
            self.eos_token = eos_token
            self.unk_token = unk_token

        self.make_vocab(json_dirs)
        counter = self.freqs.copy()
    
        min_freq = max(min_freq, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        self.itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
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

        if use_mapping:
            assert pretrained_language_model_name is not None, "Pretrained language model is required if using map for vocab"
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

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)

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
        return vec.unsqueeze(0)

    def encode_answer(self, answer: str) -> torch.Tensor:
        """ Turn a answer into a vector of indices and a question length """
        vec = torch.ones(self.max_answer_length).long() * self.padding_idx
        for i, token in enumerate([self.bos_token] + answer + [self.eos_token]):
            vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx
        return vec.unsqueeze(0)

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
        if self.vectors != other.vectors:
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

    def load_vectors(self, vectors, **kwargs):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
                fasttext.vi.300d
                phow2v.syllable.100d
                phow2v.syllable.300d
            Remaining keyword arguments: Passed to the constructor of Vectors classes.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=unk_init):
        """
        Set the vectors for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `vectors` input argument.
            vectors: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
        """
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])

class ClassificationVocab(Vocab):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711

    def __init__(self, json_dirs, max_size=None, min_freq=1, bos_token="<bos>", 
                    eos_token="<eos>", padding_token="<pad>", unk_token="<unk>",
                    pretrained_language_model_name=None, vectors=None, unk_init=unk_init,
                    vectors_cache=None, tokenizer_name: Union[str, None]=None):

        super(ClassificationVocab, self).__init__(json_dirs, max_size, min_freq, 
                                                    bos_token, eos_token, padding_token, unk_token,
                                                    pretrained_language_model_name=pretrained_language_model_name, 
                                                    vectors=vectors, 
                                                    unk_init=unk_init, 
                                                    vectors_cache=vectors_cache, 
                                                    tokenizer_name=tokenizer_name)

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
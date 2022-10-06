import torch

from data_utils.utils import is_japanese_sentence, preprocess_sentence, unk_init
from builders.word_embedding_builder import build_word_embedding
from builders.vocab_builder import META_VOCAB

from collections import defaultdict, Counter
import json
from typing import Dict, List

@META_VOCAB.register()
class Vocab(object):
    """
        Defines a vocabulary object that will be used to numericalize a field.
    """
    def __init__(self, config):

        self.tokenizer = config.VOCAB.TOKENIZER

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

        self.specials = specials

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
                for answer in ann["answers"]:
                    question = preprocess_sentence(ann["question"], self.tokenizer)
                    answer = preprocess_sentence(answer, self.tokenizer)
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

@META_VOCAB.register()
class MultilingualVocab(Vocab):
    def __init__(self, config) -> None:
        super().__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        self.max_question_length = 0
        self.max_answer_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                for answer in ann["answers"]:
                    question = ann["question"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else: # This is Vietnamese or English annotation
                        question = preprocess_sentence(ann["question"], self.tokenizer)
                        answer = preprocess_sentence(answer, self.tokenizer)
                    self.freqs.update(question)
                    self.freqs.update(answer)
                    if len(question) + 2 > self.max_question_length:
                            self.max_question_length = len(question) + 2
                    if len(answer) + 2 > self.max_answer_length:
                        self.max_answer_length = len(answer) + 2

@META_VOCAB.register()
class VlspEvjVqaVocab(MultilingualVocab):
    '''
        This vocab is designed specially for EVJVQA dataset
    '''
    
    def __init__(self, config) -> None:
        self.tokenizer = config.VOCAB.TOKENIZER

        self.padding_token = config.VOCAB.PAD_TOKEN
        self.bos_token = config.VOCAB.BOS_TOKEN
        self.eos_token = config.VOCAB.EOS_TOKEN
        self.unk_token = config.VOCAB.UNK_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV
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

        self.specials = specials

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

@META_VOCAB.register()
class ClassificationVocab(Vocab):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711

    def __init__(self, config):
        super(ClassificationVocab, self).__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        itoa = set()
        self.max_question_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"], self.tokenizer)
                for answer in ann["answers"]:
                    answer = ann["answer"]
                    answer = "_".join(answer.split())
                    self.freqs.update(question)
                    itoa.add(answer)
                if len(question) + 2 > self.max_question_length:
                        self.max_question_length = len(question) + 2

        self.itoa = {ith: answer for ith, answer in enumerate(itoa)}
        self.atoi = defaultdict()
        self.atoi.update({answer: ith for ith, answer in self.itoa.items()})
        self.total_answers = len(self.atoi)

    def encode_answer(self, answer: str) -> torch.Tensor:
        return torch.tensor([self.atoi[answer]], dtype=torch.long)

    def decode_answer(self, answer_vecs: torch.Tensor) -> List[str]:
        answers = []
        list_answers = answer_vecs.tolist()
        for answer_idx in list_answers:
            answers.append(" ".join(self.itoa[answer_idx].split("_")))

        return answers

@META_VOCAB.register()
class MultilingualClassificationVocab(ClassificationVocab):
    ''''
        This vocab is designed specially for EVJVQA dataset when treat the VQA task as classification task
    '''
    def __init__(self, config) -> None:
        super().__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        itoa = set()
        self.max_question_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                question = ann["question"]
                for answer in ann["answers"]:
                    if is_japanese_sentence(question): # This is Japanese annotation
                        question = list(question)
                    else: # This is Vietnamese or English annotation
                        question = preprocess_sentence(question, self.tokenizer)
                        answer = preprocess_sentence(answer, self.tokenizer)
                        answer = "_".join(answer)
                    itoa.add(answer)
                self.freqs.update(question)
                if len(question) + 2 > self.max_question_length:
                        self.max_question_length = len(question) + 2

        self.itoa = {ith: answer for ith, answer in enumerate(itoa)}
        self.atoi = defaultdict()
        self.atoi.update({answer: ith for ith, answer in self.itoa.items()})
        self.total_answers = len(self.atoi)

@META_VOCAB.register()
class MultiModalVocab(Vocab):
    def __init__(self, config):

        self.tokenizer = config.VOCAB.TOKENIZER

        self.padding_token = config.VOCAB.PAD_TOKEN
        self.bos_token = config.VOCAB.BOS_TOKEN
        self.eos_token = config.VOCAB.EOS_TOKEN
        self.unk_token = config.VOCAB.UNK_TOKEN
        self.img_token = config.VOCAB.IMG_TOKEN
        self.feat_token = config.VOCAB.FEAT_TOKEN
        self.box_token = config.VOCAB.BOX_TOKEN
        self.question_token = config.VOCAB.QUESTION_TOKEN
        self.answer_token = config.VOCAB.ANSWER_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.question_token, self.answer_token]
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
        self.img_idx = self.stoi[self.img_token]
        self.feat_idx = self.stoi[self.feat_token]
        self.box_idx = self.stoi[self.box_token]
        self.question_idx = self.stoi[self.question_token]
        self.answer_idx = self.stoi[self.answer_token]

        self.specials = specials

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

@META_VOCAB.register()
class MultilingualMultiModalVocab(MultiModalVocab):
    def __init__(self, config) -> None:
        super().__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        self.max_question_length = 0
        self.max_answer_length = 0
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            for ann in json_data["annotations"]:
                for answer in ann["answers"]:
                    question = ann["question"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else: # This is Vietnamese or English annotation
                        question = preprocess_sentence(ann["question"], self.tokenizer)
                        answer = preprocess_sentence(answer, self.tokenizer)
                    self.freqs.update(question)
                    self.freqs.update(answer)
                    if len(question) + 2 > self.max_question_length:
                            self.max_question_length = len(question) + 2
                    if len(answer) + 2 > self.max_answer_length:
                        self.max_answer_length = len(answer) + 2

@META_VOCAB.register()
class OcrVocab(MultiModalVocab):
    '''
        This class is designed especially for VQA with reading comprehension
    '''
    def __init__(self, config):

        self.tokenizer = config.VOCAB.TOKENIZER

        self.padding_token = config.VOCAB.PAD_TOKEN
        self.bos_token = config.VOCAB.BOS_TOKEN
        self.eos_token = config.VOCAB.EOS_TOKEN
        self.unk_token = config.VOCAB.UNK_TOKEN
        self.img_token = config.VOCAB.IMG_TOKEN
        self.feat_token = config.VOCAB.FEAT_TOKEN
        self.box_token = config.VOCAB.BOX_TOKEN
        self.ocr_token = config.VOCAB.OCR_TOKEN
        self.ocr_det_token = config.VOCAB.OCR_DET_TOKEN
        self.ocr_rec_token = config.VOCAB.OCR_REC_TOKEN
        self.question_token = config.VOCAB.QUESTION_TOKEN
        self.answer_token = config.VOCAB.ANSWER_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.ocr_token, self.ocr_det_token, self.ocr_rec_token, 
                    self.question_token, self.answer_token]
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
        self.img_idx = self.stoi[self.img_token]
        self.feat_idx = self.stoi[self.feat_token]
        self.box_idx = self.stoi[self.box_token]
        self.ocr_idx = self.stoi[self.ocr_token]
        self.ocr_det_idx = self.stoi[self.ocr_det_token]
        self.ocr_rec_idx = self.stoi[self.ocr_rec_token]
        self.question_idx = self.stoi[self.question_token]
        self.answer_idx = self.stoi[self.answer_token]

        self.specials = specials

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

    def encode_answer(self, answer: List[str], ocr_id_of: Dict[str, int]) -> torch.Tensor:
        """ Turn a answer into a vector of indices and a question length """
        vec = torch.ones(self.max_answer_length).long() * self.padding_idx
        print(answer)
        print(ocr_id_of)
        for i, token in enumerate([self.bos_token] + answer + [self.eos_token]):
            if token in ocr_id_of:
                id = ocr_id_of[token]
                print(token)
            elif token in self.stoi:
                id = self.stoi[token]
            else:
                id = self.unk_idx
            vec[i] = id
        print("+"*10)
        return vec

    def decode_answer(self, answer_vecs: torch.Tensor, ocr_token_of: Dict[int, str], join_words=True) -> List[str]:
        '''
            answer_vecs: (bs, max_length)
        '''
        answers = []
        for vec in answer_vecs:
            answer = " ".join([self.itos[idx] if self.itos[idx] not in self.specials and idx in self.itos else ocr_token_of[idx] for idx in vec.tolist()])
            if join_words:
                answers.append(answer)
            else:
                answers.append(answer.strip().split())

        return answers

@META_VOCAB.register()
class VlspVqaMultiModalVocab(MultilingualMultiModalVocab):
    def __init__(self, config) -> None:
        self.tokenizer = config.VOCAB.TOKENIZER

        self.padding_token = config.VOCAB.PAD_TOKEN
        self.bos_token = config.VOCAB.BOS_TOKEN
        self.eos_token = config.VOCAB.EOS_TOKEN
        self.unk_token = config.VOCAB.UNK_TOKEN
        self.img_token = config.VOCAB.IMG_TOKEN
        self.feat_token = config.VOCAB.FEAT_TOKEN
        self.box_token = config.VOCAB.BOX_TOKEN
        self.ocr_token = config.VOCAB.OCR_TOKEN
        self.ocr_det_token = config.VOCAB.OCR_DET_TOKEN
        self.ocr_rec_token = config.VOCAB.OCR_REC_TOKEN
        self.question_token = config.VOCAB.QUESTION_TOKEN
        self.answer_token = config.VOCAB.ANSWER_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.ocr_token, self.ocr_det_token, self.ocr_rec_token, 
                    self.question_token, self.answer_token]
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
        self.img_idx = self.stoi[self.img_token]
        self.feat_idx = self.stoi[self.feat_token]
        self.box_idx = self.stoi[self.box_token]
        self.ocr_idx = self.stoi[self.ocr_token]
        self.ocr_det_idx = self.stoi[self.ocr_det_token]
        self.ocr_rec_idx = self.stoi[self.ocr_rec_token]
        self.question_idx = self.stoi[self.question_token]
        self.answer_idx = self.stoi[self.answer_token]

        self.specials = specials

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

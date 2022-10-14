import torch

from data_utils.vocabs.vocab import Vocab
from builders.word_embedding_builder import build_word_embedding
from builders.vocab_builder import META_VOCAB

from typing import Dict, List
import itertools

@META_VOCAB.register()
class OcrVocab(Vocab):
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
        itos = specials
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
            itos.append(word)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.ocr_token, self.ocr_det_token, self.ocr_rec_token, 
                    self.question_token, self.answer_token]

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

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))

    def encode_answer(self, answer: List[str], ocr_id_of: Dict[str, int]) -> torch.Tensor:
        """ Turn a answer into a vector of indices and a question length """
        ocr_id_of = {token: idx for idx, token in enumerate(itertools.chain(*ocr_id_of))}
        vec = torch.ones(self.max_answer_length).long() * self.padding_idx
        for i, token in enumerate([self.bos_token] + answer + [self.eos_token]):
            if token in ocr_id_of:
                id = ocr_id_of[token]
            elif token in self.stoi:
                id = self.stoi[token]
            else:
                id = self.unk_idx
            vec[i] = id
        return vec

    def decode_answer(self, answer_vecs: torch.Tensor, ocr_token_of: List[List[str]], join_words=True) -> List[str]:
        '''
            answer_vecs: (bs, max_length)
        '''
        ocr_token_of = {idx: token for idx, token in enumerate(itertools.chain(*ocr_token_of))}
        answers = []
        for vec in answer_vecs:
            answer = []
            for idx in vec.tolist():
                if idx in self.specials:
                    continue
                if idx in self.itos:
                    answer.append(self.itos[idx])
                    continue
                if idx in ocr_token_of:
                    answer.append(ocr_token_of[idx])
                    continue
            answer = " ".join(answer)
            if join_words:
                answers.append(answer)
            else:
                answers.append(answer.strip().split())

        return answers
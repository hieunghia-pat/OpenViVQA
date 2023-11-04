from transformers import BertTokenizer

from builders.word_embedding_builder import build_word_embedding
from builders.vocab_builder import META_VOCAB
from data_utils.vocabs.ocr_vocab import OcrVocab
from data_utils.utils import preprocess_sentence

from typing import List, Dict
import numpy as np
import json

@META_VOCAB.register()
class BertOcrVocab(OcrVocab):
    '''
        This class is designed especially for VQA with reading comprehension
    '''
    def __init__(self, config):

        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.tokenizer = tokenizer.tokenize

        self.padding_token = tokenizer.pad_token
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.unk_token = tokenizer.unk_token
        self.img_token = config.IMG_TOKEN
        self.feat_token = config.FEAT_TOKEN
        self.box_token = config.BOX_TOKEN
        self.ocr_token = config.OCR_TOKEN
        self.ocr_det_token = config.OCR_DET_TOKEN
        self.ocr_rec_token = config.OCR_REC_TOKEN
        self.question_token = config.QUESTION_TOKEN
        self.answer_token = config.ANSWER_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])

        itos = tokenizer.ids_to_tokens
        self.stoi = {token: id for id, token in self.itos.items()}

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
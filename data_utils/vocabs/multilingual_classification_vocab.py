from data_utils.vocabs.classification_vocab import ClassificationVocab
from data_utils.utils import is_japanese_sentence, preprocess_sentence
from builders.vocab_builder import META_VOCAB

from collections import defaultdict, Counter
import json

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
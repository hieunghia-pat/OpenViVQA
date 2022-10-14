from data_utils.utils import is_japanese_sentence, preprocess_sentence
from data_utils.vocabs.vocab import Vocab
from builders.vocab_builder import META_VOCAB

from collections import Counter
import json

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

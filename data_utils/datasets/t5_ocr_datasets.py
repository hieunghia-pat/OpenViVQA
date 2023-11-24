import torch

from data_utils.datasets.ocr_datasets import OcrFeatureDataset
from data_utils.datasets.ocr_datasets import OcrDictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import re

@META_DATASET.register()
class T5OcrFeatureDataset(OcrFeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def __getitem__(self, idx: int):
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answer"]

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        answer_tokens = self.vocab.encode_answer(answer, ocr_tokens)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class T5OcrDictionaryDataset(OcrDictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]
        answers = [re.sub(r"_", "", answer) for answer in answers]

        ocr_tokens = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["ocr_texts"]]
        ocr_tokens = [self.vocab.tokenize(text) for text in ocr_tokens]
        for key in features.keys():
            features[key] = features[key].tolist()
        for idx in range(len(ocr_tokens)):
            tokens = ocr_tokens[idx]
            len_tokens = len(tokens)
            if len_tokens > 0:
                left_part = ocr_tokens[:idx]
                right_part = ocr_tokens[idx:]
                left_part.extend(tokens)
                ocr_tokens = left_part + right_part
                for key in features.keys():
                    list_features: list = features[key]
                    feature = list_features[idx]
                    for _ in range(len_tokens):
                        list_features.insert(idx, feature)
                    features[key] = list_features
        features["ocr_texts"] = ocr_tokens

        return Instance(
            **features,
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers
        )

from data_utils.datasets.ocr_datasets import OcrDictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import re

@META_DATASET.register()
class BertOcrDictionaryDataset(OcrDictionaryDataset):
    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]
        answers = [re.sub(r"\s+#+", "", answer) for answer in answers]

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

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

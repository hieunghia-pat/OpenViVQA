import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
from PIL import Image, ImageFile
from typing import Dict, List

ImageFile.LOAD_TRUNCATED_IMAGES = True

@META_DATASET.register()
class ImageQuestionDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "question": ann["question"],
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = Image.open(image_file).convert("RGB")

        question = item["question"]
        answer = item["answer"]
        answer_tokens = self.vocab.encode_answer(answer)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instance(
            question_id=idx,
            filename=image_file,
            image=image,
            question=question,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class ImageQuestionDictionaryDataset(DictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "type": ann["QA-type"],
                        "question": ann["question"],
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        
        image = Image.open(os.path.join(self.image_path, filename)).convert("RGB")
        question = item["question"]
        answers = item["answers"]

        return Instance(
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            image=image,
            question=question,
            answers=answers
        )

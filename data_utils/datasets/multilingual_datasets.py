from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from data_utils.utils import preprocess_sentence, is_japanese_sentence
from builders.dataset_builder import META_DATASET

from typing import Dict, List

@META_DATASET.register()
class MultilingualFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    for answer in ann["answers"]:
                        if is_japanese_sentence(question):
                            question = list(question)
                            answer = list(answer)
                        else:
                            question = preprocess_sentence(question, self.vocab.tokenizer)
                            answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

@META_DATASET.register()
class MultilingualDictionaryDataset(DictionaryDataset):
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
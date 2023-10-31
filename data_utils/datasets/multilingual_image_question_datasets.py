from data_utils.datasets.image_question_datasets import ImageQuestionDataset, ImageQuestionDictionaryDataset
from data_utils.utils import preprocess_sentence, is_japanese_sentence
from builders.dataset_builder import META_DATASET

from PIL import ImageFile
from typing import Dict, List

ImageFile.LOAD_TRUNCATED_IMAGES = True

@META_DATASET.register()
class MultilingualImageQuestionDataset(ImageQuestionDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        if not is_japanese_sentence(answer):
                            answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        else:
                            answer = list(answer)
                        annotation = {
                            "question": ann["question"],
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

@META_DATASET.register()
class MultilingualImageQuestionDictionaryDataset(ImageQuestionDictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answers = []
                    for answer in ann["answers"]:
                        if not is_japanese_sentence(question):
                            answer = " ".join(preprocess_sentence(answer, self.vocab.tokenizer))
                        else:
                            answer = " ".join(list(answer))
                        answers.append(answer)
                    annotations.append({
                        "question_id": ann["id"],
                        "question": ann["question"],
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    })
                    break

        return annotations
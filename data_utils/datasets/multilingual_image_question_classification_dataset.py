from data_utils.datasets.image_question_classification_dataset import ImageQuestionClassificationDataset
from data_utils.utils import preprocess_sentence, is_japanese_sentence
from builders.dataset_builder import META_DATASET

from PIL import ImageFile
from typing import Dict, List

ImageFile.LOAD_TRUNCATED_IMAGES = True

@META_DATASET.register()
class MultilingualImageQuestionClassificationDataset(ImageQuestionClassificationDataset):
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
                            answer = "_".join(answer)
                        annotation = {
                            "question": ann["question"],
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations
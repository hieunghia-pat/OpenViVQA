from data_utils.datasets.feature_classification_dataset import FeatureClassificationDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@META_DATASET.register()
class ImageQuestionClassificationDataset(FeatureClassificationDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = Image.open(image_file).convert("RGB")

        question = item["question"]
        answer = item["answer"]
        answer_tokens = self.vocab.encode_answer(answer)

        return Instance(
            question_id=idx,
            filename=image_file,
            image=image,
            question=question,
            answer=answer,
            answer_tokens=answer_tokens
        )
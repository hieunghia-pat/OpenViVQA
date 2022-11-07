from data_utils.datasets.base_dataset import BaseDataset
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
from PIL import Image, ImageFile
from typing import Dict, List

ImageFile.LOAD_TRUNCATED_IMAGES = True

@META_DATASET.register()
class ImageDataset(BaseDataset):
    # This class is designed especially for visualizing purposes
    def __init__(self, json_path: str, vocab, config) -> None:
        super(ImageDataset, self).__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    for answer in answers:
                        annotations.append({
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        })
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = Image.open(image_file)

        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])
        features = self.load_features(item["image_id"])

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            image=image,
            question=question,
            answer=answer
        )
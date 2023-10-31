from data_utils.datasets.base_dataset import BaseDataset
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from builders.dataset_builder import META_DATASET
from typing import Dict, List

@META_DATASET.register()
class FeatureClassificationDataset(BaseDataset):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711
    
    def __init__(self, json_path: str, vocab, config) -> None:
        super(FeatureClassificationDataset, self).__init__(json_path, vocab, config)

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "id": ann["id"],
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instance(
            question_id=item["id"],
            image_id=item["image_id"],
            filename=item["filename"],
            question_tokens=question,
            answer=answer,
            **features
        )

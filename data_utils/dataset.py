import torch
from torch.utils import data

from data_utils.utils import preprocess_sentence
from data_utils.vocab import ClassificationVocab, Vocab
from utils.instances import Instances

import json
import os
import numpy as np
import cv2 as cv
from typing import Dict, List, Union, Any

class BaseDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, scene_text_features_path: str=None, 
                    scene_text_threshold=None, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(BaseDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # quesion-answer pairs
        self.annotations = self.load_json(json_data)

        # image features
        self.image_features_path = image_features_path

        # scene text features
        self.scene_text_feature_path = scene_text_features_path
        self.scene_text_threshold = scene_text_threshold

    def load_json(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    annotation = {
                        "question": preprocess_sentence(ann["question"], self.vocab.tokenizer),
                        "answer": preprocess_sentence(ann["answer"], self.vocab.tokenizer),
                        "image_id": ann["image_id"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_image_feature(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        
        return features

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_feature_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]

        return features

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_feature(image_id)
        if self.scene_text_feature_path is not None:
            scene_text_features = self.load_scene_text_features(image_id)
            features = {
                **image_features,
                **scene_text_features
            }
        else:
            features = image_features

        return features

    def __getitem__(self, idx: int):
        raise NotImplementedError("Please inherit the BaseDataset class and implement the __getitem__ method")

    def __len__(self) -> int:
        return len(self.annotations)

class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, image_features_path: str, scene_text_features_path: str=None, 
                    scene_text_threshold=None, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(DictionaryDataset, self).__init__(json_path, image_features_path, scene_text_features_path,
                                                    scene_text_threshold, vocab, tokenizer_name)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = self.vocab.encode_question(item["question"])
        answer = item["answer"]

        return Instances(
            filename=filename,
            question_tokens=question,
            answer=answer,
            **features
        )

class ImageDataset(BaseDataset):
    # This class is designed especially for visualizing purposes
    def __init__(self, json_path: str, image_features_path: str, scene_text_features_path: str=None, 
                    scene_text_threshold=None, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(ImageDataset, self).__init__(json_path, image_features_path, scene_text_features_path,
                                                    scene_text_threshold, vocab, tokenizer_name)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = cv.imread(image_file)
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

        question = item["question"]
        answer = item["answer"]
        features = self.load_features(item["image_id"])

        return Instances(
            **features,
            question=question,
            answer=answer
        )

class FeatureDataset(BaseDataset):
    def __init__(self, json_path: str, image_features_path: str, scene_text_features_path: str=None, 
                    scene_text_threshold=None, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureDataset, self).__init__(json_path, image_features_path, scene_text_features_path,
                                                    scene_text_threshold, vocab, tokenizer_name)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instances(
            question_tokens=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            **features,
        )

    def __len__(self) -> int:
        return len(self.annotations)

class FeatureClassificationDataset(BaseDataset):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711
    
    def __init__(self, json_path: str, image_features_path: str, vocab: ClassificationVocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureClassificationDataset, self).__init__(json_path, image_features_path, vocab, tokenizer_name)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instances(
            question_tokens=question,
            answer_tokens=answer,
            **features
        )

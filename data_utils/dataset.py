import torch
from torch.utils import data

from data_utils.utils import preprocess_sentence, default_value
from data_utils.vocab import ClassificationVocab, Vocab

import json
import os
import numpy as np
import cv2 as cv
from typing import Dict, List, Tuple, Union
from collections import defaultdict

class DictionaryDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(DictionaryDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # set of question-answer pairs
        self.annotations = self.load_json(json_data)

        # image features or raw images
        self.image_features_path = image_features_path

    @property
    def max_question_length(self) -> int:
        if not hasattr(self, '_max_question_length'):
            self._max_question_length = max(map(len, iterables=[item["question"] for item in self.data])) + 2
        
        return self._max_question_length

    @property
    def max_answer_length(self) -> int:
        if not hasattr(self, '_max_answer_length'):
            self._max_answer_length = max(map(len, iterables=[item["answer"] for item in self.data])) + 2
        
        return self._max_answer_length

    def load_json(self, json_data: Dict) -> List[Dict]:
        filenames = {}
        for image in json_data["images"]:
            filenames[image["id"]] = image["filename"]

        annotations = []
        for ann in json_data["annotations"]:
            question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
            answer = preprocess_sentence(ann["answer"], self.vocab.tokenizer)
            question = " ".join(question)
            answer = " ".join(answer)
            annotations.append({
                "image_id": ann["image_id"],
                "filename": filenames[ann["image_id"]],
                "question": question,
                "answer": answer
            })

        return annotations
    
    @property
    def questions(self) -> List[str]:
        return [item["question"] for item in self.data]

    @property
    def answers(self) -> List[str]:
        return [item["answer"] for item in self.data]

    def load_feature(self, image_id: int) -> Tuple[np.ndarray]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]

        region_features = feature["region_features"]
        grid_feature = feature["grid_features"]
        boxes = feature["boxes"]
        # grid_size = feature["grid_size"]
        grid_size = None

        return region_features, grid_feature, boxes, grid_size

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        region_features, grid_features, boxes, grid_size = self.load_feature(image_id)
        question = item["question"]
        answer = item["answer"]

        result_dict = {
            "image_id": image_id, 
            "filename": filename, 
            "region_features": region_features, 
            "grid_features": grid_features,
            "boxes": boxes,
            "grid_size": grid_size, 
            "question": question,
            "answer": answer
        }

        returning_dict = defaultdict(default_value)
        returning_dict.update(result_dict)

        return returning_dict

    def __len__(self) -> int:
        return len(self.annotations)

class ImageDataset(DictionaryDataset):
    # This class is designed especially for visualizing purposes
    def __init__(self, json_path: str, image_path: str, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(DictionaryDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # set of question-answer pairs
        self.annotations = self.load_json(json_data)

        # image features or raw images
        self.image_path = image_path

    def load_feature(self, image_id: int) -> Tuple[np.ndarray]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]

        region_features = feature["region_features"]
        grid_feature = feature["grid_features"]
        boxes = feature["boxes"]
        # grid_size = feature["grid_size"]
        grid_size = None

        return region_features, grid_feature, boxes, grid_size

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        filename = item["filename"]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = cv.imread(image_file)
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

        question = item["question"]
        answer = item["answer"]
        region_features, grid_features, boxes, grid_size = self.load_feature(item["image_id"])

        result_dict = {
            "filename": filename,
            "image": image,
            "region_features": region_features, 
            "grid_features": grid_features,
            "boxes": boxes,
            "grid_size": grid_size,
            "question": question,
            "answer": answer
        }

        returning_dict = defaultdict(default_value)
        returning_dict.update(result_dict)

        return returning_dict

class FeatureDataset(data.Dataset):
    def __init__(self, json_path: str, image_features_path: str, vocab: Vocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = Vocab([json_path], tokenizer_name=tokenizer_name) if vocab is None else vocab

        # quesion-answer pairs
        self.annotations = self.load_json(json_data)

        # images
        self.image_features_path = image_features_path

    @property
    def max_question_length(self) -> int:
        if not hasattr(self, '_max_question_length'):
            self._max_quesiton_length = max(map(len, [item["question"] for item in self.annotations])) + 2
        
        return self._max_length

    @property
    def max_answer_length(self) -> int:
        if not hasattr(self, '_max_answer_length'):
            self._max_answer_length = max(map(len, iterables=[item["answer"] for item in self.data])) + 2
        
        return self._max_answer_length

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

    def load_feature(self, image_id: int) -> Tuple[np.ndarray]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]

        region_features = feature["region_features"]
        grid_feature = feature["grid_features"]
        boxes = feature["boxes"]
        # grid_size = feature["grid_size"]
        grid_size = None

        return region_features, grid_feature, boxes, grid_size

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        region_features, grid_features, boxes, grid_size = self.load_feature(self.annotations[idx]["image_id"])

        result_dict = {
            "region_features": region_features,
            "grid_features": grid_features,
            "boxes": boxes,
            "grid_size": grid_size,
            "question_tokens": question,
            "answer_tokens": answer,
            "shifted_right_answer_tokens": shifted_right_answer
        }

        returning_dict = defaultdict(default_value)
        returning_dict.update(result_dict)

        return returning_dict

    def __len__(self) -> int:
        return len(self.annotations)

class FeatureClassificationDataset(FeatureDataset):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711
    
    def __init__(self, json_path: str, image_features_path: str, vocab: ClassificationVocab = None, tokenizer_name: Union[str, None] = None) -> None:
        super(FeatureClassificationDataset, self).__init__(json_path, image_features_path, vocab, tokenizer_name)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])
        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(shifted_right_answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        region_features, grid_features, boxes, grid_size = self.load_feature(self.annotations[idx]["image_id"])

        result_dict = {
            "region_features": region_features,
            "grid_features": grid_features,
            "boxes": boxes,
            "grid_size": grid_size,
            "question_tokens": question,
            "answer_tokens": answer,
            "shifted_right_answer_tokens": shifted_right_answer
        }

        returning_dict = defaultdict()
        returning_dict.update(result_dict)

        return returning_dict

import torch

from .feature_classification_dataset import FeatureClassificationDataset
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
import numpy as np
from typing import Dict, List, Any

@META_DATASET.register()
class OcrClassificationDataset(FeatureClassificationDataset):
    '''
        Designed especially for LoRRA method
    '''
    
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT

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
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def load_image_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        return features

    def pad_array(self, array: np.ndarray, max_len: int, value: int = 0):
        pad_value_array = np.zeros((max_len-array.shape[0], array.shape[-1])).fill(value)
        array = np.concatenate([array, pad_value_array], axis=0)
        
        return array

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value: int = 0):
        pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value)
        tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        
        return tensor

    def pad_list(self, list: List, max_len: int, value: int = 0):
        pad_value_list = [value] * (max_len - len(list))
        list.extend(pad_value_list)

        return list

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        # select ocr features and tokens having confident score greater than a threshold
        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            else:
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature
        # get the top confident-score ocr features and tokens
        if np.array(selected_ids).sum() > self.max_scene_text:
            topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor):
                    feature = feature[topk_scores.indices]
                else:
                    feature = [feature[idx] for idx in topk_scores.indices]
                features[key] = feature
        else: # pad to the highest number of ocr tokens
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor):
                    features[key] = self.pad_tensor(feature, self.max_scene_text, 1.)
                elif isinstance(feature, np.ndarray):
                    features[key] = self.pad_array(feature, self.max_scene_text, 1.)
                elif isinstance(feature, list):
                    if isinstance(feature[0], str):
                        features[key] = self.pad_list(feature, self.max_scene_text, self.vocab.padding_token)
                    else:
                        features[key] = self.pad_list(feature, self.max_scene_text, 1.)

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_scores": features["scores"]
        }

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        features = {
            **image_features,
            **scene_text_features
        }

        return features

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        
        features = self.load_features(self.annotations[idx]["image_id"])

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"], ocr_tokens)

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            question_tokens=question,
            answer=answer,
            ocr_tokens=ocr_tokens
        )

import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instances import Instances
from builders.dataset_builder import META_DATASET

import os
import numpy as np
from typing import Dict, Any

@META_DATASET.register()
class OcrFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD

    def load_image_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        return features

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            else:
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature

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
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answer"]

        ocr_tokens = [text for text in features["ocr_texts"]] if len(features["ocr_texts"]) > 0 else [self.vocab.padding_token]
        answer_tokens = self.vocab.encode_answer(answer, ocr_tokens)
        shifted_right_answer_tokens = answer_tokens[1:] # ignore the bos token
        answer_tokens = answer_tokens[:-1] # ignore the eos token

        return Instances(
            **features,
            ocr_tokens=ocr_tokens,
            question=question,
            question_tokens=question_tokens,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class OcrDictionaryDataset(DictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD

    def load_image_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        return features

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            else:
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature

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
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]
        ocr_tokens = [text for text in features["ocr_texts"]] if len(features["ocr_texts"]) > 0 else [self.vocab.padding_token]

        return Instances(
            **features,
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            question=question,
            question_tokens=question_tokens,
            answers=answers
        )
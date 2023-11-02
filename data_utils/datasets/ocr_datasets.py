import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
import numpy as np
from typing import Dict, List, Any

@META_DATASET.register()
class OcrFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT

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
        feature_file = os.path.join(self.scene_text_features_path, f"{str(image_id)}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        # select ocr features and tokens having confident score greater than a threshold
        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            elif not isinstance(feature, int):
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature
        # get the top confident-score ocr features and tokens
        if np.array(selected_ids).sum() > self.max_scene_text:
            topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor):
                    feature = feature[topk_scores.indices]
                elif not isinstance(feature, int):
                    feature = [feature[idx] for idx in topk_scores.indices]
                features[key] = feature
        
        if len(features["det_features"]) == 0:
            det_features=self.pad_tensor(torch.zeros(1,256),self.max_scene_text,0)
        else:
            det_features=self.pad_tensor(features["det_features"],self.max_scene_text,0)
        
        if len(features["rec_features"]) ==0:
            rec_features=self.pad_tensor(torch.zeros(1,256),self.max_scene_text,0)
        else:
            rec_features=self.pad_tensor(features["rec_features"],self.max_scene_text,0)

        if len(features["fasttext_features"]) == 0:
            fasttext_features=self.pad_tensor(torch.zeros(1,300),self.max_scene_text,0)
        else:
            fasttext_features=self.pad_tensor(features["fasttext_features"],self.max_scene_text,0)

        if len(features["boxes"]) ==0 :
            boxes=self.pad_tensor(torch.zeros(1,4),self.max_scene_text,0)
        else:
            boxes=self.pad_tensor(features["boxes"],self.max_scene_text,0)

        return {
            "ocr_det_features": det_features,
            "ocr_rec_features": rec_features,
            "ocr_fasttext_features": fasttext_features,
            "ocr_texts": features["texts"],
            "ocr_boxes": boxes,
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

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        answer_tokens = self.vocab.encode_answer(answer, ocr_tokens)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
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
        self.max_scene_text = config.MAX_SCENE_TEXT

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
        feature_file = os.path.join(self.scene_text_features_path, f"{str(image_id)}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        # select ocr features and tokens having confident score greater than a threshold
        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            elif not isinstance(feature, int):
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature
        # get the top confident-score ocr features and tokens
        if np.array(selected_ids).sum() > self.max_scene_text:
            topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor):
                    feature = feature[topk_scores.indices]
                elif not isinstance(feature, int):
                    feature = [feature[idx] for idx in topk_scores.indices]
                features[key] = feature

        if len(features["det_features"]) == 0:
            det_features=self.pad_tensor(torch.zeros(1,256),self.max_scene_text,0)
        else:
            det_features=self.pad_tensor(features["det_features"],self.max_scene_text,0)
        
        if len(features["rec_features"]) ==0:
            rec_features=self.pad_tensor(torch.zeros(1,256),self.max_scene_text,0)
        else:
            rec_features=self.pad_tensor(features["rec_features"],self.max_scene_text,0)

        if len(features["fasttext_features"]) == 0:
            fasttext_features=self.pad_tensor(torch.zeros(1,300),self.max_scene_text,0)
        else:
            fasttext_features=self.pad_tensor(features["fasttext_features"],self.max_scene_text,0)

        if len(features["boxes"]) ==0 :
            boxes=self.pad_tensor(torch.zeros(1,4),self.max_scene_text,0)
        else:
            boxes=self.pad_tensor(features["boxes"],self.max_scene_text,0)

        return {
            "ocr_det_features": det_features,
            "ocr_rec_features": rec_features,
            "ocr_fasttext_features": fasttext_features,
            "ocr_texts": features["texts"],
            "ocr_boxes": boxes,
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

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        return Instance(
            **features,
            question_id=item["question_id"],
            # type=item["type"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers
        )
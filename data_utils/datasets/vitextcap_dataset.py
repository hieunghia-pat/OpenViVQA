import os
import numpy as np
import torch
import json

from data_utils.datasets.feature_dataset import FeatureDataset, BaseDataset 
from builders.task_builder import META_TASK
from utils.instance import Instance  
from data_utils.utils import preprocess_sentence
from typing import Dict, List, Any


@META_TASK.register()
class ViTextCapsDataset:
    def __init__(self, json_path: str, vocab, config) -> None:

        self.vocab = vocab

        self.image_features_path = config.FEATURE_DATASET.FEATURE_PATH.FEATURES
        self.fasttext_path = config.FEATURE_DATASET.FEATURE_PATH.FASTTEXT

        self.annotations = self.load_annotations(json_path)
        
        self.scene_text_features_path = config.FEATURE_DATASET.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.FEATURE_DATASET.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.FEATURE_DATASET.MAX_SCENE_TEXT
        
        self.fasttext_features_path = config.FEATURE_DATASET.FEATURE_PATH.FASTTEXT

    def load_annotations(self, json_path: str) -> List[Dict]:

        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

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
    
    def load_fasttext_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.fasttext_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
                
        return torch.tensor(features)

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)
        
        ocr_fasttext_features = self.load_fasttext_features(image_id)

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_fasttext_features": ocr_fasttext_features,  # Thêm đặc trưng FastText
        }
    
    def load_fasttext_features(self, image_id: int):
        feature_file = os.path.join(self.fasttext_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]
        return {'ocr_token_embeddings': feature}

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        fasttext_features = self.load_fasttext_features(image_id)
        features = {
            **image_features,
            **scene_text_features,
            **fasttext_features
        }

        return features

    def __len__(self):
        """Trả về số lượng mẫu dữ liệu trong dataset."""
        return len(self.annotations)
    
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
        answer_mask = torch.where(answer_tokens > 0, 1, 0)
        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens,
            answer_mask=answer_mask
        )
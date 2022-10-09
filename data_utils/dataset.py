import torch
from torch.utils import data

from data_utils.utils import preprocess_sentence, is_japanese_sentence
from utils.instances import Instances
from builders.dataset_builder import META_DATASET

import json
import os
import numpy as np
from PIL import Image, ImageFile
from typing import Dict, List, Any

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(BaseDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_annotations(json_data)

        # image features
        self.image_features_path = config.FEATURE_PATH.FEATURES

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        raise NotImplementedError

    def load_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)
        
        return features

    def __getitem__(self, idx: int):
        raise NotImplementedError("Please inherit the BaseDataset class and implement the __getitem__ method")

    def __len__(self) -> int:
        return len(self.annotations)

@META_DATASET.register()
class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(DictionaryDataset, self).__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "type": ann["QA-type"],
                        "question": question,
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]

        return Instances(
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            question_tokens=question_tokens,
            answers=answers,
            **features
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
        map_tokens_to_ids = {}
        ith = 0
        for text in features["ocr_texts"]:
            if text not in map_tokens_to_ids:
                map_tokens_to_ids[text] = len(self.vocab) + ith
                ith += 1
        map_ids_to_tokens = {id: token for token, id in map_tokens_to_ids.items()}

        return Instances(
            **features,
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            question_tokens=question_tokens,
            answers=answers,
            map_tokens_to_ids=map_tokens_to_ids,
            map_ids_to_tokens=map_ids_to_tokens
        )

@META_DATASET.register()
class MultilingualDictionaryDataset(DictionaryDataset):
    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "type": ann["QA-type"],
                        "question": ann["question"],
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations    

@META_DATASET.register()
class ImageQuestionDictionaryDataset(DictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "type": ann["QA-type"],
                        "question": ann["question"],
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        
        image = Image.open(os.path.join(self.image_path, filename)).convert("RGB")
        question = item["question"]
        answers = item["answers"]

        return Instances(
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            image=image,
            question=question,
            answers=answers
        )

@META_DATASET.register()
class MultilingualImageQuestionDictionaryDataset(ImageQuestionDictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answers = []
                    for answer in ann["answers"]:
                        if not is_japanese_sentence(question):
                            answer = " ".join(preprocess_sentence(answer, self.vocab.tokenizer))
                        else:
                            answer = " ".join(list(answer))
                        answers.append(answer)
                    annotations.append({
                        "question_id": ann["id"],
                        "question": ann["question"],
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    })
                    break

        return annotations

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

        return Instances(
            **features,
            image=image,
            question=question,
            answer=answer
        )

@META_DATASET.register()
class FeatureDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(FeatureDataset, self).__init__(json_path, vocab, config)

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
        question = self.vocab.encode_question(item["question"])
        map_tokens_to_ids = {}
        ith = 0
        for text in features["ocr_texts"]:
            if text not in map_tokens_to_ids:
                map_tokens_to_ids[text] = len(self.vocab) + ith
                ith += 1
        map_ids_to_tokens = {id: token for token, id in map_tokens_to_ids.items()}
        answer = self.vocab.encode_answer(item["answer"], map_tokens_to_ids)

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer

        return Instances(
            **features,
            question_tokens=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            map_tokens_to_ids=map_tokens_to_ids,
            map_ids_to_tokens=map_ids_to_tokens
        )

@META_DATASET.register()
class MultilingualFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    for answer in ann["answers"]:
                        if is_japanese_sentence(question):
                            question = list(question)
                            answer = list(answer)
                        else:
                            question = preprocess_sentence(question, self.vocab.tokenizer)
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

@META_DATASET.register()
class ImageQuestionDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

        self.image_path = config.FEATURE_PATH.IMAGE

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "question": ann["question"],
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_file = os.path.join(self.image_path, f"{item['filename']}")
        image = Image.open(image_file).convert("RGB")

        question = item["question"]
        answer = item["answer"]
        answer_tokens = self.vocab.encode_answer(answer)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instances(
            question_id=idx,
            filename=image_file,
            image=image,
            question=question,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class MultilingualImageQuestionDataset(ImageQuestionDataset):
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
                        else:
                            answer = list(answer)
                        annotation = {
                            "question": ann["question"],
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

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
                        answer = "_".join(answer)
                        annotation = {
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

        return Instances(
            question_tokens=question,
            answer_tokens=answer,
            **features
        )

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

        return Instances(
            question_id=idx,
            filename=image_file,
            image=image,
            question=question,
            answer=answer,
            answer_tokens=answer_tokens
        )

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
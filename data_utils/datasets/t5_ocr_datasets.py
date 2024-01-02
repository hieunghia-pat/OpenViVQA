import torch

from data_utils.datasets.ocr_datasets import OcrFeatureDataset
from data_utils.datasets.ocr_datasets import OcrDictionaryDataset
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import re
from typing import Dict, List

@META_DATASET.register()
class T5OcrFeatureDataset(OcrFeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def refine_ocr_features(self, ocr_tokens, dict_features):
        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = features.tolist()

        idx = 0
        while idx < len(ocr_tokens):
            tokens = ocr_tokens[idx]
            len_tokens = len(tokens)
            # shifting the current sequence to the right
            left_part = ocr_tokens[:idx]
            right_part = ocr_tokens[idx+1:]
            left_part.extend(tokens)
            ocr_tokens = left_part + right_part
            for key in dict_features:
                features: list = dict_features[key]
                feature = features[idx]
                for _ in range(len_tokens-1):
                    features.insert(idx, feature)
                dict_features[key] = features
            idx += len_tokens

        for key in dict_features:
            features = dict_features[key]
            assert len(features) == len(ocr_tokens)
            dict_features[key] = torch.Tensor(features)

        return ocr_tokens, dict_features

    def refine_obj_features(self, obj_tags, dict_features):
        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = features.tolist()

        idx = 0
        while idx < len(obj_tags):
            tags = obj_tags[idx]
            len_tags = len(tags)
            # shifting the current sequence to the right
            left_part = obj_tags[:idx]
            right_part = obj_tags[idx+1:]
            left_part.extend(tags)
            obj_tags = left_part + right_part
            for key in dict_features:
                features: list = dict_features[key]
                feature = features[idx]
                for _ in range(len_tags-1):
                    features.insert(idx, feature)
                dict_features[key] = features
            idx += len_tags

        for key in dict_features:
            features = dict_features[key]
            assert len(features) == len(obj_tags)
            dict_features[key] = torch.Tensor(features)

        return obj_tags, dict_features
    
    def append_question(self, question: List[str], obj_list: List[str], ocr_tokens: List[str]) -> List[str]:
        if len(obj_list) > 0:
            for obj in obj_list[:-1]:
                question.extend([obj, self.vocab.sep_token] if "▁" not in obj else [obj])
            question.extend([obj_list[-1]])

        if len(ocr_tokens) > 0:
            for ocr_token in ocr_tokens[:-1]:
                question.extend([ocr_token, self.vocab.sep_token] if "▁" not in ocr_token else [ocr_token])
            question.extend([ocr_tokens[-1]])

        return question

    def __getitem__(self, idx: int):
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        width = features["width"]
        height = features["height"]
        img_size = torch.Tensor((width, height, width, height))

        ocr_tokens = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["ocr_texts"]]
        ocr_tokens = [self.vocab.tokenizer(text) for text in ocr_tokens]
        relevant_ocr_keys = [
            "ocr_det_features",
            "ocr_rec_features",
            "ocr_fasttext_features",
            "ocr_boxes"
        ]
        ocr_tokens, refined_ocr_features = self.refine_ocr_features(ocr_tokens, {
            key: features[key] for key in relevant_ocr_keys
        })
        for key in relevant_ocr_keys:
            features[key] = refined_ocr_features[key]
        ocr_tokens = [text if text in self.vocab.stoi else self.vocab.padding_token for text in ocr_tokens]
        features["ocr_texts"] = self.vocab.encode_token(ocr_tokens)

        obj_list = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["object_list"]]
        obj_list = [self.vocab.tokenizer(text) for text in obj_list]
        relevant_obj_keys = [
            "region_features",
            "region_boxes"
        ]
        obj_list, refined_obj_features = self.refine_obj_features(obj_list, {
            key: features[key] for key in relevant_obj_keys
        })
        for key in relevant_obj_keys:
            features[key] = refined_obj_features[key]
        obj_list = [text if text in self.vocab.stoi else self.vocab.padding_token for text in obj_list]
        features["object_list"] = self.vocab.encode_token(obj_list)

        question = item["question"]
        question = self.append_question(question, obj_list, ocr_tokens)
        question_tokens = self.vocab.encode_question(question)
        
        answer = item["answer"]
        answer_tokens = self.vocab.encode_answer(answer)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            image_size=img_size,
            ocr_tokens=features["ocr_texts"],
            question=" ".join(question),
            question_tokens=question_tokens,
            answer=answer,
            answer_tokens=answer_tokens,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class T5OcrDictionaryDataset(OcrDictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = ["".join(answer) for answer in answers]
                    answers = [re.sub("▁", " ", answer).strip() for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "question": question,
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def refine_ocr_features(self, ocr_tokens, dict_features):
        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = features.tolist()

        idx = 0
        while idx < len(ocr_tokens):
            tokens = ocr_tokens[idx]
            len_tokens = len(tokens)
            # shifting the current sequence to the right
            left_part = ocr_tokens[:idx]
            right_part = ocr_tokens[idx+1:]
            left_part.extend(tokens)
            ocr_tokens = left_part + right_part
            for key in dict_features:
                features: list = dict_features[key]
                feature = features[idx]
                for _ in range(len_tokens-1):
                    features.insert(idx, feature)
                dict_features[key] = features
            idx += len_tokens

        for key in dict_features:
            features = dict_features[key]
            assert len(features) == len(ocr_tokens)
            dict_features[key] = torch.Tensor(features)

        return ocr_tokens, dict_features

    def refine_obj_features(self, obj_tags, dict_features):
        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = features.tolist()

        idx = 0
        while idx < len(obj_tags):
            tags = obj_tags[idx]
            len_tags = len(tags)
            # shifting the current sequence to the right
            left_part = obj_tags[:idx]
            right_part = obj_tags[idx+1:]
            left_part.extend(tags)
            obj_tags = left_part + right_part
            for key in dict_features:
                features: list = dict_features[key]
                feature = features[idx]
                for _ in range(len_tags-1):
                    features.insert(idx, feature)
                dict_features[key] = features
            idx += len_tags

        for key in dict_features:
            features = dict_features[key]
            assert len(features) == len(obj_tags)
            dict_features[key] = torch.Tensor(features)

        return obj_tags, dict_features
    
    def append_question(self, question: List[str], obj_list: List[str], ocr_tokens: List[str]) -> List[str]:
        if len(obj_list) > 0:
            for obj in obj_list[:-1]:
                question.extend([obj, self.vocab.sep_token] if "▁" not in obj else [obj])
            question.extend([obj_list[-1]])

        if len(ocr_tokens) > 0:
            for ocr_token in ocr_tokens[:-1]:
                question.extend([ocr_token, self.vocab.sep_token] if "▁" not in ocr_token else [ocr_token])
            question.extend([ocr_tokens[-1]])

        return question

    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_id = item["image_id"]
        filename = item["filename"]

        features = self.load_features(image_id)
        width = features["width"]
        height = features["height"]
        img_size = torch.Tensor((width, height, width, height))

        ocr_tokens = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["ocr_texts"]]
        ocr_tokens = [self.vocab.tokenizer(text) for text in ocr_tokens]
        relevant_ocr_keys = [
            "ocr_det_features",
            "ocr_rec_features",
            "ocr_fasttext_features",
            "ocr_boxes"
        ]
        ocr_tokens, refined_ocr_features = self.refine_ocr_features(ocr_tokens, {
            key: features[key] for key in relevant_ocr_keys
        })
        for key in relevant_ocr_keys:
            features[key] = refined_ocr_features[key]
        ocr_tokens = [text if text in self.vocab.stoi else self.vocab.padding_token for text in ocr_tokens]
        features["ocr_texts"] = self.vocab.encode_token(ocr_tokens)

        obj_list = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["object_list"]]
        obj_list = [self.vocab.tokenizer(text) for text in obj_list]
        relevant_obj_keys = [
            "region_features",
            "region_boxes"
        ]
        obj_list, refined_obj_features = self.refine_obj_features(obj_list, {
            key: features[key] for key in relevant_obj_keys
        })
        for key in relevant_obj_keys:
            features[key] = refined_obj_features[key]
        obj_list = [text if text in self.vocab.stoi else self.vocab.padding_token for text in obj_list]
        features["object_list"] = self.vocab.encode_token(obj_list)

        question = item["question"]
        question = self.append_question(question, obj_list, ocr_tokens)
        question_tokens = self.vocab.encode_question(question)
        
        answers = item["answers"]

        return Instance(
            **features,
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            image_size=img_size,
            ocr_tokens=features["ocr_texts"],
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers
        )

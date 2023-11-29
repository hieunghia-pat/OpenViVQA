import torch

from data_utils.datasets.ocr_datasets import OcrFeatureDataset
from data_utils.datasets.ocr_datasets import OcrDictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import re

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
            if len_tokens > 0:
                # shifting the current sequence to the right
                ocr_tokens[idx:] = ocr_tokens[idx+1:]
                left_part = ocr_tokens[:idx]
                right_part = ocr_tokens[idx:]
                left_part.extend(tokens)
                ocr_tokens = left_part + right_part
                for key in dict_features:
                    features: list = dict_features[key]
                    feature = features[idx]
                    for _ in range(len_tokens):
                        features.insert(idx, feature)
                        idx += 1
                    dict_features[key] = features
            else:
                idx += 1

        for key in dict_features:
            features = dict_features[key]
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
            if len_tags > 0:
                # shifting the current sequence to the right
                obj_tags[idx:] = obj_tags[idx+1:]
                left_part = obj_tags[:idx]
                right_part = obj_tags[idx:]
                left_part.extend(tags)
                obj_tags = left_part + right_part
                for key in dict_features:
                    features: list = dict_features[key]
                    feature = features[idx]
                    for _ in range(len_tags):
                        features.insert(idx, feature)
                        idx += 1
                    dict_features[key] = features
            else:
                idx += 1

        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = torch.Tensor(features)

        return obj_tags, dict_features

    def __getitem__(self, idx: int):
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answer"]
        width = features["width"]
        height = features["height"]
        img_size = (width, height)

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
        ocr_tokens = self.vocab.encode_token(ocr_tokens)
        features["ocr_texts"] = ocr_tokens

        obj_list = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["object_list"]]
        obj_list = [self.vocab.tokenizer(text) for text in obj_list]
        relevant_obj_keys = [
            "region_features",
            "region_boxes",
            "grid_features",
            "grid_boxes"
        ]
        obj_list, refined_obj_features = self.refine_obj_features(obj_list, {
            key: features[key] for key in relevant_obj_keys
        })
        for key in relevant_obj_keys:
            features[key] = refined_obj_features[key]
        obj_list = [text if text in self.vocab.stoi else self.vocab.padding_token for text in obj_list]
        obj_list = self.vocab.encode_token(obj_list)
        features["object_list"] = obj_list

        answer_tokens = self.vocab.encode_answer(answer)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            image_size=img_size,
            ocr_tokens=ocr_tokens,
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

    def refine_ocr_features(self, ocr_tokens, dict_features):
        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = features.tolist()

        idx = 0
        while idx < len(ocr_tokens):
            tokens = ocr_tokens[idx]
            len_tokens = len(tokens)
            if len_tokens > 0:
                # shifting the current sequence to the right
                ocr_tokens[idx:] = ocr_tokens[idx+1:]
                left_part = ocr_tokens[:idx]
                right_part = ocr_tokens[idx:]
                left_part.extend(tokens)
                ocr_tokens = left_part + right_part
                for key in dict_features:
                    features: list = dict_features[key]
                    feature = features[idx]
                    for _ in range(len_tokens):
                        features.insert(idx, feature)
                        idx += 1
                    dict_features[key] = features
            else:
                idx += 1

        for key in dict_features:
            features = dict_features[key]
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
            if len_tags > 0:
                # shifting the current sequence to the right
                obj_tags[idx:] = obj_tags[idx+1:]
                left_part = obj_tags[:idx]
                right_part = obj_tags[idx:]
                left_part.extend(tags)
                obj_tags = left_part + right_part
                for key in dict_features:
                    features: list = dict_features[key]
                    feature = features[idx]
                    for _ in range(len_tags):
                        features.insert(idx, feature)
                        idx += 1
                    dict_features[key] = features
                idx += len_tags
            else:
                idx += 1

        for key in dict_features:
            features = dict_features[key]
            dict_features[key] = torch.Tensor(features)

        return obj_tags, dict_features

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        width = item["width"]
        height = item["height"]
        img_size = (width, height)

        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]
        answers = [re.sub(r"_", "", answer) for answer in answers]

        ocr_tokens = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["ocr_texts"]]
        ocr_tokens = [self.vocab.tokenizer(text) for text in ocr_tokens]
        relevant_ocr_keys = [
            "ocr_det_features",
            "ocr_rec_features",
            "ocr_fasttext_features",
            "ocr_boxes"
        ]
        refined_ocr_features = self.refine_ocr_features(ocr_tokens, {
            key: features[key] for key in relevant_ocr_keys
        })
        for key in relevant_ocr_keys:
            features[key] = refined_ocr_features[key]
        ocr_tokens = [text if text in self.vocab.stoi else self.vocab.padding_token for text in ocr_tokens]
        ocr_tokens = self.vocab.encode_token(ocr_tokens)
        features["ocr_texts"] = ocr_tokens

        obj_list = [text if text.strip() != "" else 
                      self.vocab.padding_token for text in features["object_list"]]
        obj_list = [self.vocab.tokenizer(text) for text in obj_list]
        relevant_obj_keys = [
            "region_features",
            "region_boxes",
            "grid_features",
            "grid_boxes"
        ]
        refined_obj_features = self.refine_obj_features(obj_list, {
            key: features[key] for key in relevant_obj_keys
        })
        for key in relevant_obj_keys:
            features[key] = refined_obj_features[key]
        obj_list = [text if text in self.vocab.stoi else self.vocab.padding_token for text in obj_list]
        obj_list = self.vocab.encode_token(obj_list)
        features["object_list"] = obj_list

        return Instance(
            **features,
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            image_size=img_size,
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers
        )

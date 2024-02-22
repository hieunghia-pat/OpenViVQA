import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET
import scipy.spatial.distance as distance
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
        self.max_scene_text = config.MAX_SCENE_TEXT
        self.sort_type=config.SORT_TYPE

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
    
    def convert_to_polygon(self, bbox, text):
        x1, y1, x2, y2 = bbox
        return [f'{text}',[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    def sorting_bounding_box(self, data):
        points=[self.convert_to_polygon(data['boxes'][i],data['texts'][i]) for i in range(len(data['boxes']))]
        points = list(map(lambda x:[x[0],x[1][0],x[1][2]],points))
        points_sum = list(map(lambda x: [x[0],x[1],sum(x[1]),x[2][1]],points))
        x_y_cordinate = list(map(lambda x: x[1],points_sum))
        final_sorted_list = []
        while True:
            try:
                new_sorted_text = []
                initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][2])][0]
                threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
                threshold_value = (threshold_value/2) + 500
                del points_sum[initial_value_A[0]]
                del x_y_cordinate[initial_value_A[0]]
                # print(threshold_value)
                A = [initial_value_A[1][1]]
                K = list(map(lambda x:[x,abs(x[1]-initial_value_A[1][1][1])],x_y_cordinate))
                K = [[count,i]for count,i in enumerate(K)]
                K = [i for i in K if i[1][1] <= threshold_value]
                sorted_K = list(map(lambda x:[x[0],x[1][0]],sorted(K,key=lambda x:x[1][1])))
                B = []
                points_index = []
                for tmp_K in sorted_K:
                    points_index.append(tmp_K[0])
                    B.append(tmp_K[1])
                dist = distance.cdist(A,B)[0]
                d_index = [i for i in sorted(zip(dist,points_index), key=lambda x:x[0])]
                new_sorted_text.append(initial_value_A[1][0])

                index = []
                for j in d_index:
                    new_sorted_text.append(points_sum[j[1]][0])
                    index.append(j[1])
                for n in sorted(index, reverse=True):
                    del points_sum[n]
                    del x_y_cordinate[n]
                final_sorted_list.append(new_sorted_text)
            except Exception as e:
                # print(e)
                break

        combined_list = [item for sublist in final_sorted_list for item in sublist]
        new_index= [data['texts'].index(item) for item in combined_list]
        return combined_list, new_index

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        if self.sort_type=='score':
            selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                    feature = feature[selected_ids]
                elif not isinstance(feature, int):
                    feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
                features[key] = feature
            
            if np.array(selected_ids).sum() > self.max_scene_text:
                topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
                for key, feature in features.items():
                    if isinstance(feature, torch.Tensor):
                        feature = feature[topk_scores.indices]
                    elif not isinstance(feature, int):
                        feature = [feature[idx] for idx in topk_scores.indices]
                    features[key] = feature
                    
        if self.sort_type=='top-left bottom-right':
            if len(features['texts'])>1:
                features['texts'], new_ids=self.sorting_bounding_box(features)
                for key, feature in features.items():
                    if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                        feature = feature[new_ids]
                    elif not isinstance(feature, int):
                        feature = [feature[idx] for idx, new_id in enumerate(new_ids) if new_id]
                    features[key] = feature

        if self.sort_type is not None and self.sort_type not in ['score', 'top-left bottom-right']:
            raise ValueError("Invalid sort_type. Must be either 'score' or 'top-left bottom-right' or None ")
        
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_tensor(feature, self.max_scene_text, 0)
            elif isinstance(feature, np.ndarray):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_array(feature, self.max_scene_text, 0)
            elif isinstance(feature, list):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_list(feature, self.max_scene_text, self.vocab.padding_token)
            features[key] = feature

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_fasttext_features": features["fasttext_features"],
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
        self.sort_type=config.SORT_TYPE

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
    
    def convert_to_polygon(self, bbox, text):
        x1, y1, x2, y2 = bbox
        return [f'{text}',[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    def sorting_bounding_box(self, data):
        points=[self.convert_to_polygon(data['boxes'][i],data['texts'][i]) for i in range(len(data['boxes']))]
        points = list(map(lambda x:[x[0],x[1][0],x[1][2]],points))
        points_sum = list(map(lambda x: [x[0],x[1],sum(x[1]),x[2][1]],points))
        x_y_cordinate = list(map(lambda x: x[1],points_sum))
        final_sorted_list = []
        while True:
            try:
                new_sorted_text = []
                initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][2])][0]
                threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
                threshold_value = (threshold_value/2) + 500
                del points_sum[initial_value_A[0]]
                del x_y_cordinate[initial_value_A[0]]
                # print(threshold_value)
                A = [initial_value_A[1][1]]
                K = list(map(lambda x:[x,abs(x[1]-initial_value_A[1][1][1])],x_y_cordinate))
                K = [[count,i]for count,i in enumerate(K)]
                K = [i for i in K if i[1][1] <= threshold_value]
                sorted_K = list(map(lambda x:[x[0],x[1][0]],sorted(K,key=lambda x:x[1][1])))
                B = []
                points_index = []
                for tmp_K in sorted_K:
                    points_index.append(tmp_K[0])
                    B.append(tmp_K[1])
                dist = distance.cdist(A,B)[0]
                d_index = [i for i in sorted(zip(dist,points_index), key=lambda x:x[0])]
                new_sorted_text.append(initial_value_A[1][0])

                index = []
                for j in d_index:
                    new_sorted_text.append(points_sum[j[1]][0])
                    index.append(j[1])
                for n in sorted(index, reverse=True):
                    del points_sum[n]
                    del x_y_cordinate[n]
                final_sorted_list.append(new_sorted_text)
            except Exception as e:
                # print(e)
                break

        combined_list = [item for sublist in final_sorted_list for item in sublist]
        new_index= [data['texts'].index(item) for item in combined_list]
        return combined_list, new_index

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]

        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        if self.sort_type=='score':
            selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                    feature = feature[selected_ids]
                elif not isinstance(feature, int):
                    feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
                features[key] = feature
            
            if np.array(selected_ids).sum() > self.max_scene_text:
                topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
                for key, feature in features.items():
                    if isinstance(feature, torch.Tensor):
                        feature = feature[topk_scores.indices]
                    elif not isinstance(feature, int):
                        feature = [feature[idx] for idx in topk_scores.indices]
                    features[key] = feature
                    
        if self.sort_type=='top-left bottom-right':
            if len(features['texts'])>1:
                features['texts'], new_ids=self.sorting_bounding_box(features)
                for key, feature in features.items():
                    if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                        feature = feature[new_ids]
                    elif not isinstance(feature, int):
                        feature = [feature[idx] for idx, new_id in enumerate(new_ids) if new_id]
                    features[key] = feature

        if self.sort_type is not None and self.sort_type not in ['score', 'top-left bottom-right']:
            raise ValueError("Invalid sort_type. Must be either 'score' or 'top-left bottom-right' or None ")
        
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_tensor(feature, self.max_scene_text, 0)
            elif isinstance(feature, np.ndarray):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_array(feature, self.max_scene_text, 0)
            elif isinstance(feature, list):
                if len(features[key])>self.max_scene_text:
                    features[key]=features[key][:self.max_scene_text]
                else:
                    features[key] = self.pad_list(feature, self.max_scene_text, self.vocab.padding_token)
            features[key] = feature
        
        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_fasttext_features": features["fasttext_features"],
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

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        return Instance(
            **features,
            question_id=item["question_id"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers
        )
import torch

from data_utils.datasets.ocr_datasets import OcrFeatureDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
import numpy as np
from typing import Dict, Any
import scipy.spatial.distance as distance

@META_DATASET.register()
class SceneTextVqaDataset(OcrFeatureDataset):
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
    
    def filter_scene_text_features(self, features: dict[str, torch.FloatTensor]) -> dict[str, torch.FloatTensor]:
        # select ocr features and tokens having confident score greater than a threshold
        selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                feature = feature[selected_ids]
            elif isinstance(feature, list):
                feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
            features[key] = feature
        # get the top confident-score ocr features and tokens
        if np.array(selected_ids).sum() > self.max_scene_text:
            topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor):
                    feature = feature[topk_scores.indices]
                elif isinstance(feature, list):
                    feature = [feature[idx] for idx in topk_scores.indices]
                features[key] = feature

        return features
    
    def convert_to_polygon(self, bbox, text):
        x1, y1, x2, y2 = bbox
        return [f'{text}',[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
    
    def sort_scene_text_features(self, 
                                 features: dict[str, torch.FloatTensor]) -> dict[str, torch.FloatTensor]:
        points=[self.convert_to_polygon(features['boxes'][i], features['texts'][i]) 
                for i in range(len(features['boxes']))]
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
        new_index= [features['texts'].index(item) for item in combined_list]

        for key in features:
            features[key] = features[key][new_index]
        
        return features

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        features = self.sort_scene_text_features(features)

        return {
            f"ocr_{key}": value for key, value in features.values()
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
        question_tokens, question_mask = self.vocab.encode_question(question)

        answer = item["answer"]
        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]
        answer_tokens, answer_mask = self.vocab.encode_answer(answer, ocr_tokens)
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
            question_mask=question_mask,
            answer=answer,
            answer_tokens=answer_tokens,
            answer_mask=answer_mask
        )

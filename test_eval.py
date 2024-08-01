from configs.utils import get_config
from models.stacmr import VSRN
from data_utils.datasets.vitextcap_dataset import ViTextCapsDataset
import pickle
# Load vocabulary
import os
import numpy as np
import torch
from typing import Dict, Any

from data_utils.datasets.feature_dataset import FeatureDataset, BaseDataset
from utils.instance import Instance
from data_utils.utils import preprocess_sentence
from typing import Dict, List
import json
from configs.utils import get_config
from tasks.training_stacmr import TrainingStacMR
class ViTextCapsDataset:
    def __init__(self, json_path: str, vocab, config) -> None:
        """
        Khởi tạo ViTextCapsDataset.

        Args:
            json_path (str): Đường dẫn đến file JSON chứa annotations.
            vocab (object): Đối tượng vocabulary để xử lý từ vựng.
            config (object): Đối tượng chứa các cấu hình.
        """
        self.vocab = vocab

        # Đường dẫn đến các đặc trưng hình ảnh và văn bản OCR
        self.image_features_path = config.DATASET.get("IMAGE_FEATURES_PATH", 'data\\object_features')
        self.scene_text_features_path = config.DATASET.get("SCENE_TEXT_FEATURES_PATH", 'data\\ocr_features')
        self.fasttext_path = config.DATASET.get("FASTTEXT_PATH", 'data\\fasttext')
        # Ngưỡng và giới hạn cho OCR
        self.scene_text_threshold = config.DATASET.get("SCENE_TEXT_THRESHOLD", 0.0)
        self.max_scene_text = config.DATASET.get("MAX_SCENE_TEXT", 100)

        # Tải các annotations
        self.annotations = self.load_annotations(json_path)

    def load_annotations(self, json_path: str) -> List[Dict]:
        """
        Tải các annotations từ file JSON.

        Args:
            json_path (str): Đường dẫn đến file JSON.

        Returns:
            List[Dict]: Danh sách các annotations.
        """
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

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"]
        }

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
        return len(self.annotations)

    def load_fasttext_features(self, image_id: int):
        feature_file = os.path.join(self.fasttext_path, f"{image_id}.npy")
        feature = np.load(feature_file, allow_pickle=True)[()]
        return {'ocr_token_embeddings': feature}


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



import pickle
# Load vocabulary
vocab = pickle.load(open(r"data\\vocab.bin", "rb"))
config_file = r'OpenViVQA\\configs\\stacmr.yaml'
config = get_config(config_file)
dataset = loader = ViTextCapsDataset('data\\dev.json',
                           vocab=vocab,
                           config=config)

import itertools
import evaluation
import tqdm
from builders.vocab_builder import build_vocab

vocab = build_vocab(config.DATASET.VOCAB)
device = 'cuda'
epoch=0
def evaluate_metrics(model, dataloader):
    model.eval()
    gens = {}
    gts = {}
    with tqdm(total=len(dataloader)) as pbar:
        for it, items in enumerate(dataloader):
            items = items.to(device)
            with torch.no_grad():
                # results = self.model(items)
                outputs = model(obj_boxes=items['grid_boxes'].squeeze(),
                                            obj_features=items['grid_features'],
                                            ocr_boxes=items['boxes'],
                                            ocr_token_embeddings=items['ocr_token_embeddings'],
                                            ocr_rec_features=items['rec_features'],
                                            ocr_det_features=items['det_features'],
                                            caption_tokens=items['answer_tokens'].squeeze(),
                                            caption_masks=items['answer_mask'].squeeze(),
                                            mode='inference')
            # outs = results["scores"].argmax(dim=-1)
            outs = outputs['predicted_token']
            answers_gt = items.answers
            answers_gen = vocab.decode_answer(outs.contiguous(),
                                                items.ocr_tokens,
                                                join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gens['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    scores, _ = evaluation.compute_scores(gts, gens)

    return scores

from data_utils.utils import collate_fn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
            dataset=loader,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn
        )
device = 'cuda'
model = VSRN(config.MODEL)
model.to(device)
model.eval()
it = 0
gens = {}
gts = {}
items = next(iter(train_dataloader))
items = items.to(device)
with torch.no_grad():
    # results = self.model(items)
    outputs = model(obj_boxes=items['grid_boxes'].squeeze().to(device),
                    obj_features=items['grid_features'].to(device),
                    ocr_boxes=items['ocr_boxes'].to(device),
                    ocr_token_embeddings=items['ocr_token_embeddings'].to(device),
                    ocr_rec_features=items['ocr_rec_features'].to(device),
                    ocr_det_features=items['ocr_det_features'].to(device),
                    caption_tokens=items['answer_tokens'].squeeze().to(device),
                    caption_masks=items['answer_mask'].squeeze().to(device),
                    mode='inference')
    
# outs = results["scores"].argmax(dim=-1)
outs = outputs['predicted_token']
answers_gt = items.answer
answers_gen = vocab.decode_answer(outs.contiguous(),
                                  items.ocr_tokens,
                                  join_words=False)
for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
    gens['%d_%d' % (it, i)] = [gen_i, ]
    gts['%d_%d' % (it, i)] = gts_i

scores, _ = evaluation.compute_scores(gts, gens)
print(scores)
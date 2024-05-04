import torch

from typing import Dict, List

from data_utils.datasets.ocr_datasets import OcrDataset
from utils.instance import Instance
from data_utils.utils import preprocess_sentence
from builders.dataset_builder import META_DATASET

@META_DATASET.register()
class PretrainedOcrDataset(OcrDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        question = ann["question"]
                        answer = answer
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
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        width = features["width"]
        height = features["height"]
        img_size = torch.Tensor((width, height, width, height))
        object_list = features["object_list"]
        if object_list == []:
            object_list = [self.vocab.padding_token]
        object_tag_ids = self.vocab.encode_tokens(object_list)

        question = item["question"]
        ocr_tokens = features["ocr_texts"]
        if ocr_tokens == []:
            ocr_tokens = [self.vocab.padding_token]
        ocr_token_ids = self.vocab.encode_tokens(ocr_tokens)

        context = " ".join(ocr_tokens)
        question_tokens, question_mask = self.vocab.encode_question(question, context) # add ocr texts as the context for question

        answer = item["answer"]
        answer_tokens, answer_mask = self.vocab.encode_answer(answer)

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            image_size=img_size,
            tags=object_tag_ids,
            ocr_tokens=ocr_token_ids,
            question=question,
            input_ids=question_tokens,
            attention_mask=question_mask,
            answer=answer,
            labels=answer_tokens,
            decoder_attention_mask=answer_mask
        )

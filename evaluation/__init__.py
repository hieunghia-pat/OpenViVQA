from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from typing import List
from .tokenizer import PTBTokenizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def compute_metrics(predicted: List[int], true: List[int]):
    acc = accuracy_score(true, predicted)
    pre = precision_score(true, predicted, average="macro", zero_division=0)
    recall = recall_score(true, predicted, average="macro", zero_division=0)
    f1 = f1_score(true, predicted, average="macro")

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": recall,
        "F1": f1
    }
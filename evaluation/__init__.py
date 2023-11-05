from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .accuracy import Accuracy
from .f1 import F1
from .precision import Precision
from .recall import Recall
import string

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Accuracy(), Precision(), Recall(), F1())
    all_score = {}
    all_scores = {}
    normalized_gts = {key: [normalize_text(value[0])] for key, value in gts.items()}
    normalized_gen = {key: [normalize_text(value[0])] for key, value in gen.items()}
    for metric in metrics:
        score, scores = metric.compute_score(normalized_gts, normalized_gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

from saved_models.metrics.evaluate_metrics import Recall
from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .f1 import F1
from .precision import Precision

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Precision(), Recall(), F1())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

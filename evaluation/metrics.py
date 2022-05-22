import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from data_utils.vocab import Vocab

class Metrics(object):
    def __init__(self, vocab: Vocab=None):
        self.vocab = vocab

    def get_scores(self, predicted, true):
        """ Compute the accuracies, precision, recall and F1 score for a batch of predictions and answers """

        predicted = self.vocab._decode_answer(predicted)
        true = self.vocab._decode_answer(true)

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

    def evaluate(self, net, test_dataset, tracker, prefix=''):
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

        loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=configuration.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=configuration.data_workers)

        tq = tqdm(loader, desc='{}'.format(prefix), ncols=0)
        acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
        pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

        for v, q, a, q_len in tq:
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()
            q_len = q_len.cuda()

            out = net(v, q, q_len)
            scores = self.get_scores(out.cpu(), a.cpu())

            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(0.0), accuracy=fmt(acc_tracker.mean.value), 
                            precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))

        return {
            "accuracy": acc_tracker.mean.value,
            "precision": pre_tracker.mean.value,
            "recall": rec_tracker.mean.value,
            "F1": f1_tracker.mean.value
        }
from sklearn.metrics import f1_score
import numpy as np

class F1:
    @classmethod
    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        res = {key: value.split() for key, value in res.items()}
        micro_scores = []
        macro_scores = []
        for key in res:
            micro_scores.append(f1_score(gts[key], res[key], average="micro", zero_division=0))
            macro_scores.append(f1_score(gts[key], res[key], average="macro", zero_division=0))

        micro_scores = np.array(micro_scores)
        macro_scores = np.array(macro_scores)

        return (micro_scores, micro_scores.mean()), (macro_scores, macro_scores.mean())
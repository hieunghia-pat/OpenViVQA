from sklearn.metrics import accuracy_score
import numpy as np

class Accuracy:
    @classmethod
    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        res = {key: value.split() for key, value in res.items()}
        scores = []
        for key in res:
            scores.append(accuracy_score(gts[key], res[key]))

        scores = np.array(scores)
        scores = np.array(scores)

        return scores, scores.mean()
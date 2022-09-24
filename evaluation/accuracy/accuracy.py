from sklearn.metrics import accuracy_score
import numpy as np

class Accuracy:
    def compute_score(self, gts, res):
        """
        Main function to compute Accuracy score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: accuracy (float) : computed Accuracy score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            g = gts[key]
            if len(r) > len(g):
                r = r[:len(g)]
            else:
                r = r + ["<pad>"]*(len(g) - len(r))
            scores.append(accuracy_score(g, r))

        scores = np.array(scores)

        return scores, scores.mean()

    def __str__(self) -> str:
        return "Accuracy"
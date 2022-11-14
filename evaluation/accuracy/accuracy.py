import numpy as np

class Accuracy:
    def compute_score(self, gts, res):
        """
        Main function to compute Accuracy score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: accuracy (float) : computed accuracy score for the corpus
        """
        res = {key: value[0] for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            scores_per_res = []
            for gt in gts[key]:
                scores_per_res.append(int(r == gt))
            score = np.array(scores_per_res).mean()
            scores.append(score)

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "Accuracy"

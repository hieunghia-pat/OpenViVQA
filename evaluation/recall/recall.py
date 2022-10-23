import numpy as np

class Recall:
    def compute_score(self, gts, res):
        """
        Main function to compute recall score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: accuracy (float) : computed Accuracy score for the corpus
        """
        res = {key: value[0].split() for key, value in res.items()}
        scores = []
        for key in res:
            r = res[key]
            scores_per_res = []
            for gt in gts[key]:
                gt = gt.split()
                # if either the prediction or the truth is no-answer then recall = 1 if they agree, 0 otherwise
                if len(r) == 0 or len(gt) == 0:
                    scores_per_res.append(int(r == gt))
                else:
                    common_tokens = set(r) & set(gt)
                    # if there are no common tokens then recall = 0
                    if len(common_tokens) == 0:
                        scores_per_res.append(0)
                    else:
                        scores_per_res.append(len(common_tokens)/len(gt))

            scores_per_res = np.array(scores_per_res).mean()
            scores.append(scores_per_res)

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "Recall"
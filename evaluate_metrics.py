import json
import os
import re
import string
import sys
import numpy as np
import copy
import math
from collections import defaultdict
from typing import Dict, List, Any

def is_japanese_sentence(text: str):
    # REFERENCE UNICODE TABLES: 
    # http:#www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
    # http:#www.tamasoft.co.jp/en/general-info/unicode.html
    #
    # TEST EDITOR:
    # http:#www.gethifi.com/tools/regex
    #
    # UNICODE RANGE : DESCRIPTION
    # 
    # 3000-303F : punctuation
    # 3040-309F : hiragana
    # 30A0-30FF : katakana
    # FF00-FFEF : Full-width roman + half-width katakana
    # 4E00-9FAF : Common and uncommon kanji
    # 
    # Non-Japanese punctuation/formatting characters commonly used in Japanese text
    # 2605-2606 : Stars
    # 2190-2195 : Arrows
    # u203B     : Weird asterisk thing
    pattern = r"[\u3000-\u303F]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\uFF00-\uFFEF]|[\u4E00-\u9FAF]|[\u2605-\u2606]|[\u2190-\u2195]|\u203B"
    return re.search(pattern, text) is not None

def is_vietnamese_sentence(text: str):
    pattern = r"[áàảãạúùủũụýỳỷỹỵíìỉĩịóòỏõọốồổỗộớờởỡợéèẻẽẹếềểễệđ]"
    return re.search(pattern, text) is not None

def normalize_answer(s, is_japanese: False):
    if is_japanese: # if the answer is Japanese then treat each string as tokens
        return list(s)
    else: # else normalize the Vietnamese and English answer, lower text, remove punctuation and articles
        def remove_punc(text):
            exclude = set(string.punctuation)
            text = re.sub("、", "", text)
            text = re.sub("。", "", text)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        
        return remove_punc(lower(s)).split()

# compute accuracy function
def compute_f1(a_gold: Dict[Any, str], a_pred: Dict[Any, str]):
    gts = {}
    res = {}
    for key in a_gold:
        answer = a_gold[key]
        gts[key] = normalize_answer(a_gold[key], is_japanese_sentence(answer))
        res[key] = normalize_answer(a_pred[key], is_japanese_sentence(answer))
    
    f1 = F1()
    score = f1.compute_score(gts, res)

    return score

# compute avg. BLEU score
def compute_avg_bleu(a_gold, a_pred):
    gts = {}
    res = {}
    for key in a_gold:
        answer = a_gold[key]
        gts[key] = [" ".join(normalize_answer(a_gold[key], is_japanese_sentence(answer)))]
        res[key] = [" ".join(normalize_answer(a_pred[key], is_japanese_sentence(answer)))]

    bleu = Bleu()
    scores, _ = bleu.compute_score(gts, res)

    return np.array(scores).mean()

class Precision:
    def compute_score(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]):
        assert(gts.keys() == res.keys()), "gts and res must have exactly the same keys"
        assert isinstance(gts, dict), "gts must be a dict where values are lists of strings"
        assert isinstance(res, dict), "res must be a dict where values are lists of strings"
        scores = []
        for key in gts:
            gt = gts[key]
            r = res[key]
            common = set(gt) & set(r)
            scores.append(len(common) / len(set(r)))

        return np.array(scores).mean()

class Recall:
    def compute_score(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]):
        assert(gts.keys() == res.keys()), "gts and res must have exactly the same keys"
        assert isinstance(gts, dict), "gts must be a dict where values are lists of strings"
        assert isinstance(res, dict), "res must be a dict where values are lists of strings"
        scores = []
        for key in gts:
            gt = gts[key]
            r = res[key]
            common = set(gt) & set(r)
            scores.append(len(common) / len(set(gt)))

        return np.array(scores).mean()

class F1:
    def precision(self, gt: List[str], r: List[str]) -> float:
        common = set(gt) & set(r)
        return len(common) / len(set(r))

    def recall(self, gt: List[str], r: List[str]) -> float:
        common = set(gt) & set(r)
        return len(common) / len(set(gt))

    def compute(self, gt: List[str], r: List[str]) -> float:
        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(r) == 0 or len(gt) == 0:
            return int(r == gt)

        precision = self.precision(gt, r)
        recall = self.recall(gt, r)

        if precision == 0 or recall == 0:
            return 0

        f1 = 2*precision*recall / (precision+recall)

        return f1

    def compute_score(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]):
        assert isinstance(gts, dict), "gts must be a dict where values are lists of strings"
        assert isinstance(res, dict), "res must be a dict where values are lists of strings"
        assert(gts.keys() == res.keys()), "gts and res must have exactly the same keys"

        scores = []
        for key in gts:
            gt = gts[key]
            r = res[key]
            scores.append(self.compute(gt, r))

        return np.array(scores).mean()

def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return (len(words), counts)


def cook_refs(refs, eff=None, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)

    ## lhuang: N.B.: leave reflen computaiton to the very end!!

    ## lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return (reflen, maxcounts)


def cook_test(test, ref_tuple, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''

    testlen, counts = precook(test, n, True)
    reflen, refmaxcounts = ref_tuple

    result = {}

    # Calculate effective reference sentence length.

    if eff == "closest":
        result["reflen"] = min((abs(l - testlen), l) for l in reflen)[1]
    else:  ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result

class BleuScorer(object):
    """Bleu scorer.
    """

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"

    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

        self._score = None  ## need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        '''
        return (bleu, len_ratio) pair
        '''

        return self.fscore(option=option), self.ratio(option=option)

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''

        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None  ## need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):

        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:  ## need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small)  ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list

class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)

        return score, scores

    def __str__(self):
        return 'BLEU'


if __name__ == "__main__":
    try:
        [_, input_dir, output_dir] = sys.argv

        with open(os.path.join(input_dir, 'ground_truth.json')) as f:
            ground_truth = json.load(f)
        
        with open(os.path.join(input_dir, 'results.json')) as f:
            results = json.load(f)

        f1 = compute_f1(ground_truth, results)
        bleu = compute_avg_bleu(ground_truth, results)

        with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
            output_file.write("F1: {:f}\nBLEU: {:f}".format(f1, bleu))

    except Exception as e:
        with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
            output_file.write(str(e))
        raise
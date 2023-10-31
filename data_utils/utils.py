import torch
import re
from typing import List

from utils.instance import Instance, InstanceList

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return lambda s: s 
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("Please install PyVi package. "
                  "See the docs at https://github.com/trungtv/pyvi for more information.")
    elif tokenizer == "spacy":
        try:
            from spacy.lang.vi import Vietnamese
            return Vietnamese()
        except ImportError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
    elif tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            # before using vncorenlp, please run this command in your terminal:
            # vncorenlp -Xmx500m data_utils/vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -annotators wseg &
            annotator = VnCoreNLP(address="http://127.0.0.1", port=9000, max_heap_size='-Xmx500m')

            def tokenize(s: str):
                words = annotator.tokenize(s)[0]
                return " ".join(words)

            return tokenize
        except ImportError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise
        except AttributeError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise

def preprocess_sentence(sentence: str, tokenizer: str):
    sentence = sentence.lower()
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    tokenizer = get_tokenizer(tokenizer)
    sentence = tokenizer(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens

def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def unk_init(token, dim):
    '''
        For default:
            + <pad> is 0
            + <sos> is 1
            + <eos> is 2
            + <unk> is 3
    '''

    if token in ["<pad>", "<p>"]:
        return torch.zeros(dim)
    if token in ["<sos>", "<bos>", "<s>"]:
        return torch.ones(dim)
    if token in ["<eos>", "</s>"]:
        return torch.ones(dim) * 2
    return torch.ones(dim) * 3

def default_value():
    return None

def collate_fn(samples: List[Instance]):
    return InstanceList(samples)

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
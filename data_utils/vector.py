import torch
import os
import logging
from tqdm import tqdm
import gzip
import tarfile
from urllib.request import urlretrieve
from data_utils.utils import reporthook
import zipfile
from functools import partial

logger = logging.getLogger(__name__)

def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim

class Vectors(object):
    def __init__(self, name, cache=None, url=None, unk_init=None, max_vectors=None):
        """
        Args:

            name: name of the file that contains the vectors
            cache: directory for cached vectors
            url: url for download if vectors not found in cache
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and returns a Tensor of the same size
            max_vectors (int): this can be used to limit the number of
                pre-trained vectors loaded.
                Most pre-trained vector sets are sorted
                in the descending order of word frequency.
                Thus, in situations where the entire set doesn't fit in memory,
                or is not needed for another reason, passing `max_vectors`
                can limit the size of the loaded set.
        """

        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = unk_init
        self.cache(name, cache, url=url, max_vectors=max_vectors)


    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            if self.unk_init is None:
                return torch.Tensor.zero_(torch.Tensor(self.dim))
            else:
                return self.unk_init(token, self.dim)

    def cache(self, name, cache, url=None, max_vectors=None):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            logger.info("Loading vectors from {}".format(path))
            ext = os.path.splitext(path)[1][1:]
            if ext == 'gz':
                open_file = gzip.open
            else:
                open_file = open

            vectors_loaded = 0
            with open_file(path, 'rb') as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines

                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

                for line in tqdm(f, total=max_vectors):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split(b" ")

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning("Skipping token {} with 1-dimensional "
                                       "vector {}; likely a header".format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            "Vector for token {} has {} dimensions, but previously "
                            "read vectors have {} dimensions. All vectors must have "
                            "the same number of dimensions.".format(word, len(entries),
                                                                    dim))
                    try:
                        if isinstance(word, bytes):
                            word = word.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                    vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

    def __len__(self):
        return len(self.vectors)

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.

        Args:
            tokens: a token or a list of tokens. if `tokens` is a string,
                returns a 1-D tensor of shape `self.dim`; if `tokens` is a
                list of strings, returns a 2-D tensor of shape=(len(tokens),
                self.dim).
            lower_case_backup : Whether to look up the token in the lower case.
                If False, each token in the original case will be looked up;
                if True, each token in the original case will be looked up first,
                if not found in the keys of the property `stoi`, the token in the
                lower case will be looked up. Default: False.

        Examples:
            >>> examples = ['chip', 'baby', 'Beautiful']
            >>> vec = text.vocab.GloVe(name='6B', dim=50)
            >>> ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
        """
        to_reduce = False

        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        if not lower_case_backup:
            indices = [self[token] for token in tokens]
        else:
            indices = [self[token] if token in self.stoi
                       else self[token.lower()]
                       for token in tokens]

        vecs = torch.stack(indices)
        return vecs[0] if to_reduce else vecs

class PhoW2V(Vectors):
    url = {
        "word2vec_vi_syllables_100dims": "https://public.vinai.io/word2vec_vi_syllables_100dims.zip",
        "word2vec_vi_syllables_300dims": "https://public.vinai.io/word2vec_vi_syllables_300dims.zip",
        "word2vec_vi_words_100dims": "https://public.vinai.io/word2vec_vi_words_100dims.zip",
        "word2vec_vi_words_300dims": "https://public.vinai.io/word2vec_vi_words_300dims.zip",
    }

    def __init__(self, name, **kwargs):
        url = self.url[name]
        name = '{}.txt'.format(name)
        super(PhoW2V, self).__init__(name, url=url, **kwargs)

class ViFastText(Vectors):

    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz'

    def __init__(self, **kwargs):
        url = self.url_base
        name = os.path.basename(url)
        super(ViFastText, self).__init__(name, url=url, **kwargs)

pretrained_aliases = {
    "fasttext.vi.300d": partial(ViFastText),
    "phow2v.syllable.100d": partial(PhoW2V, name="word2vec_vi_syllables_100dims"),
    "phow2v.syllable.300d": partial(PhoW2V, name="word2vec_vi_syllables_300dims"),
    "phow2v.word.100d": partial(PhoW2V, name="word2vec_vi_words_100dims"),
    "phow2v.word.300d": partial(PhoW2V, name="word2vec_vi_words_300dims")
}
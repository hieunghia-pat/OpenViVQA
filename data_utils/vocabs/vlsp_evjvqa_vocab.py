import torch

from data_utils.vocabs.multilingual_vocab import MultilingualVocab
from builders.word_embedding_builder import build_word_embedding
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class VlspEvjVqaVocab(MultilingualVocab):
    '''
        This vocab is designed specially for EVJVQA dataset
    '''
    
    def __init__(self, config) -> None:
        self.tokenizer = config.TOKENIZER

        self.padding_token = config.PAD_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

        self.word_embeddings = None
        if config.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))
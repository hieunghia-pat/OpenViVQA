from data_utils.vocabs.vocab import Vocab
from builders.word_embedding_builder import build_word_embedding
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class MultiModalVocab(Vocab):
    def __init__(self, config):

        self.tokenizer = config.VOCAB.TOKENIZER

        self.padding_token = config.VOCAB.PAD_TOKEN
        self.bos_token = config.VOCAB.BOS_TOKEN
        self.eos_token = config.VOCAB.EOS_TOKEN
        self.unk_token = config.VOCAB.UNK_TOKEN
        self.img_token = config.VOCAB.IMG_TOKEN
        self.feat_token = config.VOCAB.FEAT_TOKEN
        self.box_token = config.VOCAB.BOX_TOKEN
        self.question_token = config.VOCAB.QUESTION_TOKEN
        self.answer_token = config.VOCAB.ANSWER_TOKEN

        self.make_vocab([
            config.JSON_PATH.TRAIN,
            config.JSON_PATH.DEV,
            config.JSON_PATH.TEST
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.question_token, self.answer_token]
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

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token, self.img_token,
                    self.feat_token, self.box_token, self.question_token, self.answer_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]
        self.img_idx = self.stoi[self.img_token]
        self.feat_idx = self.stoi[self.feat_token]
        self.box_idx = self.stoi[self.box_token]
        self.question_idx = self.stoi[self.question_token]
        self.answer_idx = self.stoi[self.answer_token]

        self.word_embeddings = None
        if config.VOCAB.WORD_EMBEDDING is not None:
            self.load_word_embeddings(build_word_embedding(config))
from data_utils.vocab import Vocab
from models.modules.containers import Module
from models.modules.beam_search import BeamSearch
from structures.instances import Instances

class BaseTransformer(Module):
    def __init__(self, vocab: Vocab):
        super(BaseTransformer, self).__init__()

        self.max_len = vocab.max_answer_length
        self.eos_idx = vocab.eos_idx

        self.register_state('enc_features', None)
        self.register_state('enc_padding_mask', None)

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def encoder_forward(self, input_features: Instances):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        raise NotImplementedError

    def beam_search(self, input_features, beam_size: int, out_size=1, return_probs=False, **kwargs):
        # get features from input
        self.enc_features, self.enc_padding_mask = self.encoder_forward(input_features)

        bs = BeamSearch(self, self.max_len, self.eos_idx, beam_size)

        return bs.apply(out_size, return_probs, **kwargs)

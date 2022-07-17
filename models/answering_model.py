from models.modules.containers import Module
from models.modules.beam_search import BeamSearch

class AnsweringModel(Module):
    def __init__(self):
        super(AnsweringModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def beam_search(self, max_len, eos_idx, beam_size, out_size=1, return_probs=False, **visual_inputs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(out_size=out_size, return_pobs=return_probs, **visual_inputs)

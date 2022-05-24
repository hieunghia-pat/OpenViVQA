import torch
from torch import distributions
from typing import Tuple

from models import utils
from models.modules.containers import Module
from models.modules.beam_search import BeamSearch
from data_utils.feature import Feature

class AnsweringModel(Module):
    def __init__(self):
        super(AnsweringModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        raise NotImplementedError

    def test(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        device = utils.get_device(visual)
        outputs = []
        log_probs = []

        mask = torch.ones((b_s,), device=device)
        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                log_probs_t = self.step(t, out, visual, None, mode='feedback', **kwargs)
                out = torch.max(log_probs_t, -1)[1]
                mask = mask * (out.squeeze(-1) != eos_idx).float()
                log_probs.append(log_probs_t * mask.unsqueeze(-1).unsqueeze(-1))
                outputs.append(out)

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def sample_rl(self, visual: utils.TensorOrSequence, max_len: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        outputs = []
        log_probs = []

        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                out = self.step(t, out, visual, None, mode='feedback', **kwargs)
                distr = distributions.Categorical(logits=out[:, 0])
                out = distr.sample().unsqueeze(1)
                outputs.append(out)
                log_probs.append(distr.log_prob(out).unsqueeze(1))

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def beam_search(self, visual: Feature, linguistic: Feature, max_len: int, eos_idx: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(visual, linguistic, out_size, return_probs, **kwargs)

import torch
from torch import nn

from models.modules.containers import Module
from models.modules.beam_search import BeamSearch
from utils.instance import Instance

class BaseTransformer(Module):
    def __init__(self, config, vocab):
        super(BaseTransformer, self).__init__()

        self.vocab = vocab
        self.max_len = vocab.max_answer_length
        self.eos_idx = vocab.eos_idx
        self.d_model = config.D_MODEL

        self.register_state('encoder_features', None)
        self.register_state('encoder_padding_mask', None)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder_forward(self, input_features: Instance):
        raise NotImplementedError

    def forward(self, input_features: Instance):
        raise NotImplementedError

    def step(self, t, prev_output):
        bs = self.encoder_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.encoder_features.device)
        else:
            it = prev_output

        output = self.decoder(
            answer_tokens=it,
            encoder_features=self.encoder_features,
            encoder_attention_mask=self.encoder_padding_mask
        )

        return output

    def beam_search(self, input_features: Instance, batch_size: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        beam_search = BeamSearch(model=self, max_len=self.max_len, eos_idx=self.eos_idx, beam_size=beam_size, 
                            b_s=batch_size, device=self.device)

        with self.statefulness(batch_size):
            self.encoder_features, self.encoder_padding_mask = self.encoder_forward(input_features)
            output =  beam_search.apply(out_size, return_probs, **kwargs)

        return output

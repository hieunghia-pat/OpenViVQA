import torch
from torch import nn

from data_utils.vocab import Vocab
from models.modules.containers import Module
from models.modules.beam_search import BeamSearch
from utils.instances import Instances

class BaseUniqueTransformer(Module):
    def __init__(self, config, vocab: Vocab):
        super(BaseUniqueTransformer, self).__init__()

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

    def embed_features(self, input_features: Instances):
        raise NotImplementedError

    def append_answer(self, joint_features, joint_attention_mask, answer_tokens):
        raise NotImplementedError
    
    def forward(self, input_features: Instances):
        raise NotImplementedError

    def step(self, t, prev_output):
        bs = self.encoder_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.encoder_features.device)
        else:
            it = prev_output

        self.encoder_features, self.encoder_padding_mask = self.append_answer(self.encoder_features, self.encoder_padding_mask, it)
        out = self.encoder(Instances(
            features=self.encoder_features,
            features_attention_mask=self.encoder_padding_mask
        ))

        return out

    def beam_search(self, input_features: Instances, batch_size: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        beam_search = BeamSearch(model=self, max_len=self.max_len, eos_idx=self.eos_idx, beam_size=beam_size, 
                            b_s=batch_size, device=self.device)

        with self.statefulness(batch_size):
            self.encoder_features, self.encoder_padding_mask = self.embed_features(input_features)
            self.join_feature_len = self.encoder_features.shape[1]
            output =  beam_search.apply(out_size, return_probs, **kwargs)

        return output
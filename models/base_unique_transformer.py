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

    def append_answer(self, joint_features, joint_masks, answer_tokens, maps_ids_to_tokens, maps_tokens_to_features):
        answer_features, (answer_padding_mask, answer_sequential_mask) = self.text_embedding(answer_tokens)
        answer_self_attention_mask = torch.logical_or(answer_padding_mask, answer_sequential_mask) # (bs, 1, answer_len, answer_len)
        a_tokens = torch.ones((answer_tokens.shape[0], answer_tokens.shape[1])).long().to(answer_tokens.device) * self.vocab.answer_idx
        a_embedded, _ = self.text_embedding(a_tokens)
        answer_features += a_embedded

        joint_features_len = joint_features.shape[1]
        answer_len = answer_features.shape[1]
        joint_features = torch.cat([joint_features, answer_features], dim=1)
        joint_padding_mask, joint_attention_mask = joint_masks
        joint_padding_mask = torch.cat([joint_padding_mask, answer_padding_mask], dim=-1)
        # joint features cannot see the answer features
        batch_size = joint_features.shape[0]
        joint_features_mask_answer = torch.ones((batch_size, 1, joint_features_len, answer_len)).bool().to(joint_features.device) # (bs, 1, joint_features_len, answer_len)
        joint_attention_mask = torch.cat([joint_attention_mask, joint_features_mask_answer], dim=-1) # (bs, 1, joint_features_len, joint_features_len + answer_len)
        # answer tokens can attend to all joint features
        answer_attend_joint_features = torch.zeros((batch_size, 1, answer_len, joint_features_len)).bool().to(answer_features.device) # (bs, 1, answer_len, joint_features_len)
        answer_attend_joint_features = torch.cat([answer_attend_joint_features, answer_self_attention_mask], dim=-1) # (bs, 1, answer_len, joint_features_len + answer_len)
        joint_attention_mask = torch.cat([joint_attention_mask, answer_attend_joint_features], dim=-2) # (bs, 1 , joint_features_len + answer_len, joint_features_len + answer_len)

        return joint_features, (joint_padding_mask, joint_attention_mask)
    
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
import torch
from torch import nn
from torch.nn import init, functional as F
from .utils import generate_padding_mask
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.base_transformer import BaseTransformer
from builders.model_builder import META_ARCHITECTURE
from builders.vision_embedding_builder import build_vision_embedding
from builders.decoder_builder import build_decoder
from utils.instance import Instance

class CoAttention(nn.Module):
    def __init__(self, config):
        super(CoAttention, self).__init__()
        self.v_conv = nn.Linear(config.D_VISION, config.D_MODEL, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(config.D_LANGUAGE, config.D_MODEL)
        self.x_conv = nn.Linear(config.D_MODEL, config.GLIMPSES)

        self.drop = nn.Dropout(config.DROPOUT)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = q.unsqueeze(1).expand_as(v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

class TextProcessor(nn.Module):
    def __init__(self, config, vocab):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(len(vocab.stoi), config.D_EMBEDDING, padding_idx=0)
        if vocab.word_embeddings is not None:
            self.embedding.from_pretrained(vocab.vectors, padding_idx=vocab.stoi["<pad>"])
        self.drop = nn.Dropout(config.DROPOUT)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=config.D_EMBEDDING,
                            hidden_size=config.D_MODEL,
                            num_layers=1,
                            batch_first=True)

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        _, (_, c) = self.lstm(tanhed)
        return c.squeeze(0)

@META_ARCHITECTURE.register()
class IterativeSAAA(BaseTransformer):
    """ 
        Re-implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering" (https://arxiv.org/abs/1704.03162).
    """

    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.vision = build_vision_embedding(config.VISION_PROCESSOR)
        self.text = TextProcessor(config.TEXT_PROCESSOR, vocab)
        self.attention = CoAttention(config.ATTENTION)

        self.fusion = PositionWiseFeedForward(config.MULTIMODAL_FUSION)
        self.norm = nn.LayerNorm(config.MULTIMODAL_FUSION.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab)

        self.padding_idx = vocab.padding_idx

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def apply_attention(self, input, attention):
        """ Apply any number of attention maps over the input. """
        input = input.unsqueeze(1).permute(0, 1, -1, -2) # [n, 1, dim, s]
        attention = attention.permute(0, -1, 1) # [n, g, s]
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, dim, s]
        weighted = weighted.sum(dim=1).permute(0, -1, 1) # [n, s, dim]
        
        return weighted

    def encoder_forward(self, input_features: Instance):
        v = input_features.region_features
        q = input_features.question_tokens

        v, v_padding_mask = self.vision(v)
        q = self.text(q)
        q_padding_mask = generate_padding_mask(q.unsqueeze(dim=1), padding_idx=self.padding_idx)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = self.apply_attention(v, a)

        q = q.unsqueeze(dim=1)
        combined = torch.cat([v, q], dim=1)
        combined_mask = torch.cat([v_padding_mask, q_padding_mask], dim=-1)
        combined = self.fusion(combined)
        combined = combined.masked_fill(combined_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
        combined = self.norm(combined)

        return combined, combined_mask

    def forward(self, input_features: Instance):
        combined, combined_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        out = self.decoder(
            answer_tokens=answer_tokens,
            encoder_features=combined,
            encoder_attention_mask=combined_mask
        )

        return F.log_softmax(out, dim=-1)
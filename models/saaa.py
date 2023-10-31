import torch
from torch import nn
from torch.nn import init, functional as F

from models.base_classification import BaseClassificationModel
from builders.model_builder import META_ARCHITECTURE
from builders.vision_embedding_builder import build_vision_embedding
from builders.text_embedding_builder import build_text_embedding
from utils.instance import InstanceList

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

# class TextProcessor(nn.Module):
#     def __init__(self, config, vocab):
#         super(TextProcessor, self).__init__()
#         self.embedding = nn.Embedding(len(vocab.stoi), config.D_EMBEDDING, padding_idx=0)
#         if vocab.word_embeddings is not None:
#             self.embedding.from_pretrained(vocab.vectors, padding_idx=vocab.stoi["<pad>"])
#         self.drop = nn.Dropout(config.DROPOUT)
#         self.tanh = nn.Tanh()
#         self.lstm = nn.LSTM(input_size=config.D_EMBEDDING,
#                             hidden_size=config.D_MODEL,
#                             num_layers=1,
#                             batch_first=True)

#         self._init_lstm(self.lstm.weight_ih_l0)
#         self._init_lstm(self.lstm.weight_hh_l0)
#         self.lstm.bias_ih_l0.data.zero_()
#         self.lstm.bias_hh_l0.data.zero_()

#         init.xavier_uniform_(self.embedding.weight)

#     def _init_lstm(self, weight):
#         for w in weight.chunk(4, 0):
#             init.xavier_uniform_(w)

#     def forward(self, q):
#         embedded = self.embedding(q)
#         tanhed = self.tanh(self.drop(embedded))
#         _, (_, c) = self.lstm(tanhed)
#         return c.squeeze(0)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

@META_ARCHITECTURE.register()
class SAAA(BaseClassificationModel):
    """ 
        Re-implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering" (https://arxiv.org/abs/1704.03162).
    """

    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.vision = build_vision_embedding(config.VISION_PROCESSOR)
        self.text = build_text_embedding(config.TEXT_PROCESSOR, vocab)
        self.attention = CoAttention(config.ATTENTION)

        self.classifier = Classifier(
            in_features=config.ATTENTION.GLIMPSES * config.ATTENTION.D_VISION + config.ATTENTION.D_LANGUAGE,
            mid_features=1024,
            out_features=vocab.total_answers,
            drop=0.5,
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def apply_attention(self, input, attention):
        """ Apply any number of attention maps over the input. """
        n = input.shape[0]

        # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
        input = input.view(n, 1, -1, self.d_model).permute(0, 1, 3, 2) # [n, 1, d_model, s]
        attention = attention.permute(0, -1, 1)
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, v, s]
        weighted_mean = weighted.sum(dim=-1) # [n, g, v]
        
        return weighted_mean.view(n, -1)

    def forward(self, input_features: InstanceList):
        v = input_features.region_features
        q = input_features.question_tokens

        v, _ = self.vision(v)
        q, _ = self.text(q)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = self.apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        out = self.classifier(combined)

        return F.log_softmax(out, dim=-1)

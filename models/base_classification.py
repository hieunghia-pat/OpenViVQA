from torch import nn

from utils.instance import Instance

class BaseClassificationModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.d_model = config.D_MODEL

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_features: Instance):
        raise NotImplementedError

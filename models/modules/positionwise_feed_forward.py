import torch
from torch import nn
from torch.nn import functional as F

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, config) -> None:
        super(PositionWiseFeedForward, self).__init__()

        d_model = config.D_MODEL
        d_ff = config.D_FF
        dropout = config.DROPOUT

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input) -> torch.Tensor:
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out)

        return out
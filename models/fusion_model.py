from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model*2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model*2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return self.fc2(self.dropout(x))

class FusionModel(nn.Module):
    def __init__(self, d_model, dropout):
        super(FusionModel, self).__init__()

        self.mlp_v = MLP(d_model, dropout)
        self.mlp_q = MLP(d_model, dropout)

        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_q = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, v_encoded, q_encoded):
        v_attended = F.softmax(self.mlp_v(v_encoded), dim=1)
        q_attended = F.softmax(self.mlp_q(q_encoded), dim=1)

        v_features = (v_attended * v_encoded).sum(dim=1)
        q_features = (q_attended * q_encoded).sum(dim=1)

        z = self.norm(self.fc_v(v_features) + self.fc_q(q_features))

        return z
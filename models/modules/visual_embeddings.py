from torch import nn

from models.utils import generate_padding_mask

class VisualEmbedding(nn.Module):
    def __init__(self, config):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(config.D_FEATURES, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, visual):
        masks = generate_padding_mask(visual, padding_idx=0).to(visual.device)

        visual = self.proj(visual)
        visual = self.dropout(visual)

        return visual, masks

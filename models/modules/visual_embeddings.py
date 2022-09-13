from torch import nn

from builders.build_visual_embedding import META_VISUAL_EMBEDDING
from models.utils import generate_padding_mask

@META_VISUAL_EMBEDDING.register()
class VisualEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(VisualEmbedding, self).__init__()

        self.proj = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual):
        masks = generate_padding_mask(visual, padding_idx=0).to(visual.device)

        visual = self.proj(visual)
        visual = self.dropout(visual)

        return visual, masks

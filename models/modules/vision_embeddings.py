from torch import nn

from builders.vision_embedding_builder import META_VISION_EMBEDDING
from models.utils import generate_padding_mask

@META_VISION_EMBEDDING.register()
class VisionEmbedding(nn.Module):
    def __init__(self, config):
        super(VisionEmbedding, self).__init__()

        self.proj = nn.Linear(config.D_FEATURES, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, features):
        masks = generate_padding_mask(features, padding_idx=0).to(features.device)

        features = self.proj(features)
        features = self.dropout(features)

        return features, masks

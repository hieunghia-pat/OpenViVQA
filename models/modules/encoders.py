import torch
from torch.nn import functional as F
from torch import nn
from models.utils import generate_padding_mask, clones, box_relational_embedding, get_combine_masks
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention, ScaledDotProductAttention, AugmentedGeometryScaledDotProductAttention, AugmentedMemoryScaledDotProductAttention
from models.modules.embeddings import SinusoidPositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                    can_be_stateful=False, use_aoa=False, attention_module=None, **attention_module_kwargs):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        can_be_stateful=can_be_stateful,
                                        attention_module=attention_module,
                                        **attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask, padding_mask=None, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)
        if padding_mask is not None:
            ff = ff.masked_fill(padding_mask, value=0)

        return ff

class Encoder(nn.Module):
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level_output=False,
                 identity_map_reordering=False, use_aoa=False, **attention_module_kwargs):
        super(Encoder, self).__init__()

        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.pos_embedding = SinusoidPositionalEmbedding(d_model, normalize=True)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=ScaledDotProductAttention,
                                                  **attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.multi_level_output = multi_level_output

    def forward(self, features, **kwargs):
        padding_masks = generate_padding_mask(features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

        features = F.relu(self.fc(features))
        features = self.dropout(features)
        out = self.layer_norm(features)

        if self.multi_level_output:
            outs = []
        pos_embedding = self.pos_embedding(out)
        for layer in self.layers:
            out = out + pos_embedding
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_masks)
            if self.multi_level_output:
                outs.append(out.unsqueeze(1))

        if self.multi_level_output:
            outs = torch.cat(outs, dim=1)
            return outs, padding_masks, pos_embedding
        
        return out, padding_masks, pos_embedding

class AugmentedMemoryEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level_output=False,
                 identity_map_reordering=False, use_aoa=False, **attention_module_kwargs):
        super(AugmentedMemoryEncoder, self).__init__()

        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.pos_embedding = SinusoidPositionalEmbedding(d_model, normalize=True)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=AugmentedMemoryScaledDotProductAttention,
                                                  **attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.multi_level_output = multi_level_output

    def forward(self, features, **kwargs):
        padding_masks = generate_padding_mask(features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

        features = F.relu(self.fc(features))
        features = self.dropout(features)
        out = self.layer_norm(features)

        if self.multi_level_output:
            outs = []
        pos_embedding = self.pos_embedding(out)
        for layer in self.layers:
            out = layer(queries=out + pos_embedding, keys=out + pos_embedding, values=out, attention_mask=padding_masks)
            if self.multi_level_output:
                outs.append(out.unsqueeze(1))

        if self.multi_level_output:
            outs = torch.cat(outs, dim=1)
            return outs, padding_masks, pos_embedding
        
        return out, padding_masks, pos_embedding

class AugmentedGeometryEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level_output=False,
                 identity_map_reordering=False, use_aoa=False, trignometric_embedding=True, **attention_module_kwargs):
        super(AugmentedGeometryEncoder, self).__init__()

        self.trignometric_embedding = trignometric_embedding
        if trignometric_embedding:
            self.d_g = d_model // h
        else:
            self.d_g = 4

        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_gs = clones(nn.Linear(self.d_g, 1), h)

        self.pos_embedding = SinusoidPositionalEmbedding(d_model, normalize=True)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=AugmentedGeometryScaledDotProductAttention,
                                                  **attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.multi_level_output = multi_level_output

        self.init_weights()

    def init_weights(self):
        for fc_g in self.fc_gs:
            nn.init.xavier_uniform_(fc_g.weight)

        for fc_g in self.fc_gs:
            nn.init.constant_(fc_g.bias, 0)

    def forward(self, features, boxes, **kwargs):
        padding_masks = generate_padding_mask(features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

        features = F.relu(self.fc(features))
        features = self.dropout(features)
        out = self.layer_norm(features)

        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        if self.multi_level_output:
            outs = []

        pos_embedding = self.pos_embedding(out)
        for layer in self.layers:
            out = layer(queries=out + pos_embedding, keys=out + pos_embedding, values=out, 
                        relative_geometry_weights=relative_geometry_weights, 
                        attention_mask=padding_masks)
            if self.multi_level_output:
                outs.append(out.unsqueeze(1))

        if self.multi_level_output:
            outs = torch.cat(outs, dim=1)
            return outs, padding_masks, pos_embedding
        
        return out, padding_masks, pos_embedding

class DualCollaborativeLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level_output=False,
                 identity_map_reordering=False, use_aoa=False, trignometric_embedding=True, **attention_module_kwargs):
        super(DualCollaborativeLevelEncoder, self).__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.trignometric_embedding = trignometric_embedding
        if trignometric_embedding:
            self.d_g = d_model // h
        else:
            self.d_g = 4

        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_nrom_grid = nn.LayerNorm(self.d_model)

        self.fc_gs = clones(nn.Linear(self.d_g, 1), h)

        self.pos_embedding = SinusoidPositionalEmbedding(d_model, normalize=True)

        # Attention on regions
        self.layers_region = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                         identity_map_reordering=identity_map_reordering,
                                                         use_aoa = use_aoa,
                                                         attention_module=AugmentedGeometryScaledDotProductAttention,
                                                         **attention_module_kwargs)
                                            for _ in range(N)])

        # Attention on grids
        self.layers_grid = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       use_aoa = use_aoa,
                                                       attention_module=AugmentedGeometryScaledDotProductAttention,
                                                       **attention_module_kwargs)
                                          for _ in range(N)])

        # Cross Attention between regions and grids
        self.region2grid = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                               identity_map_reordering=identity_map_reordering,
                                               use_aoa = use_aoa,
                                               attention_module=AugmentedGeometryScaledDotProductAttention,
                                               **attention_module_kwargs)
                                          for _ in range(N)])

        # Cross Attention between grids and regions
        self.grid2region = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                               identity_map_reordering=identity_map_reordering,
                                               use_aoa = use_aoa,
                                               attention_module=AugmentedGeometryScaledDotProductAttention,
                                               **attention_module_kwargs)
                                          for _ in range(N)])

        self.padding_idx = padding_idx

        # Whether using multi level output in encoder head.
        self.multi_level_output = multi_level_output

        self.init_weights()

    def init_weights(self):
        for fc_g in self.fc_gs:
            nn.init.xavier_uniform_(fc_g.weight)

        for fc_g in self.fc_gs:
            nn.init.constant_(fc_g.bias, 0)

    def forward(self, region_features, region_boxes, grid_features, grid_boxes, **kwargs):
        out_region = F.relu(self.fc_region(region_features))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)

        out_grid = F.relu(self.fc_grid(grid_features))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_nrom_grid(out_grid)

        torch.cat([out_region, out_grid], dim=1)

        n_regions = region_boxes.shape[1]
        n_grids = grid_boxes.shape[1]
        grid_size = int(n_grids**0.5) # default is 7

        boxes = torch.cat([region_boxes, grid_boxes], dim=1) # (bs, n_regions + n_grids, 4)
        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        region_padding_masks = generate_padding_mask(region_features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)
        grid_padding_masks = generate_padding_mask(grid_features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)
        region2grid_padding_masks = []
        region2grid_padding_masks = get_combine_masks(region_boxes, grid_size).unsqueeze(1) # (bs, 1, n_regions, n_grids)
        grid2region_padding_masks = region2grid_padding_masks.permute(0, 1, 3, 2) # (bs, 1, n_grids, n_regions)
        region2all_padding_masks = torch.cat([region_padding_masks, region2grid_padding_masks], dim=-1) # (bs, 1, n_regions, n_regions + n_grids)
        grid2all_padding_masks = torch.cat([grid2region_padding_masks, grid_padding_masks], dim=-1) # (bs, 1, n_grids, n_regions + n_grids)

        region_pos_embedding = self.pos_embedding(out_region)
        grid_pos_embedding = self.pos_embedding(out_grid)
        region_grid_pos_embedding = self.pos_embedding(out)

        if self.multi_level_output:
            outs = []
        for l_region, l_grid, l_r2g, l_g2r in zip(self.layers_region, self.layers_grid, 
                                                    self.region2grid, self.grid2region):
            # self-attention on region feature
            out_region = out_region + region_pos_embedding
            out_region = l_region(queries=out_region, values=out_region, keys=out_region, 
                                    relative_geometry_weights=relative_geometry_weights[:, :, :n_regions, :n_regions],
                                    attention_mask=region_padding_masks)

            #self-attention on grid feature
            out_grid = out_grid + grid_pos_embedding
            out_grid = l_grid(queries=out_grid, values=out_grid, keys=out_grid,
                                relative_geometry_weights=relative_geometry_weights[:, :, n_regions:, n_regions:],
                                attention_mask=grid_padding_masks)

            # prepare the combined output
            out_combined = torch.cat([out_region, out_grid], dim=1)
            out_combined = out_combined + region_grid_pos_embedding

            # cross self-attention between regions and grids
            out_region = out_region + region_pos_embedding
            out_region = l_r2g(queries=out_region, keys=out_combined, values=out_combined, 
                                relative_geometry_weights=relative_geometry_weights[:, :, :n_regions, :],
                                attention_mask=region2all_padding_masks)

            # cross self-attention between grids and regions
            out_grid = out_grid + grid_pos_embedding
            out_grid = l_g2r(queries=out_grid, keys=out_combined, values=out_combined, 
                                relative_geometry_weights=relative_geometry_weights[:, :, n_regions:, :],
                                attention_mask=grid2all_padding_masks)

            # Concat
            out = torch.cat([out_region, out_grid], dim=1)

            if self.multi_level_output:
                outs.append(out.unsqueeze(1))

        # If 'multi_level_output' is applied.
        if self.multi_level_output:
            outs = torch.cat(outs, dim=1)

        padding_mask = torch.cat([region_padding_masks, grid_padding_masks], dim=-1)

        # If 'multi_level_output' is applied.
        if self.multi_level_output:
            return outs, padding_mask, region_pos_embedding
        
        return out, padding_mask, region_pos_embedding

Encoders = {
    "encoder": Encoder,
    "augmented-memory-encoder": AugmentedMemoryEncoder,
    "augmented-geometry-encoder": AugmentedGeometryEncoder,
    "dlct-encoder": DualCollaborativeLevelEncoder
}

def get_encoder(vocab, config):
    encoder = Encoders[config.model.transformer.encoder.module]

    return encoder(N=config.model.nlayers, padding_idx=vocab.padding_idx, d_in=config.model.d_feature, 
                    d_model=config.model.d_model, d_k=config.model.d_k, d_v=config.model.d_v,
                    d_ff=config.model.d_ff, dropout=config.model.dropout,
                    **config.model.transformer.encoder.args)
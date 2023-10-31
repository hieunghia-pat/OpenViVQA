import torch
from torch import nn
from torch.nn import functional as F
from data_utils.types import *
import copy

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s

def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s

def positional_embedding(input, d_model) -> torch.Tensor:
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32)
    out = positional_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def generate_padding_mask(sequences: TensorOrNone, padding_idx: int) -> torch.BoolTensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    if sequences is None:
        return None

    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_idx*__seq.shape[-1])).long() * -10e4 # (b_s, seq_len)
    return mask.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

def generate_sequential_mask(seq_len: int) -> torch.BoolTensor:
    '''
        Mask out subsequent positions
    '''
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) * -10e4 # (seq_len, seq_len)

    return subsequent_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

def generate_self_attention_masks(padding_masks: torch.Tensor, sequential_masks: torch.Tensor):
    padding_masks = padding_masks != 0
    sequential_masks = sequential_masks != 0
    self_attention_masks = torch.logical_or(padding_masks, sequential_masks).long() * -10e4
    
    return self_attention_masks

def get_relative_pos(x, norm_len):
    x = x.view(-1, 1)
    return  x / norm_len

def get_grids_position(grid_size):
    w, h = grid_size
    # record the pos of each grid according to the form of region box
    x = torch.arange(0, w).float().cuda()
    y = torch.arange(0, h).float().cuda()

    px_min = x.view(-1, 1).expand(-1, w).contiguous().view(-1)
    py_min = y.view(1, -1).expand(h, -1).contiguous().view(-1)

    px_max = px_min + 1
    py_max = py_min + 1

    # scale pos into the range (0 ~ 1)
    rpx_min = get_relative_pos(px_min, w)
    rpy_min = get_relative_pos(py_min, h)

    rpx_max = get_relative_pos(px_max, w)
    rpy_max = get_relative_pos(py_max, h)

    boxes = torch.cat([rpx_min, rpy_min, rpx_max, rpy_max], dim=-1) # (n, 4)

    return boxes

def box_relational_embedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, chunks=4, dim=-1) # each tensor has dimension of (batch_size, max_nr_bounding_boxes, 1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1, -1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(f_g.device)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
        
    return embedding # (batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, dim_g)

import torch
from typing import Union, Sequence

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]
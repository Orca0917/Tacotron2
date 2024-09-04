from typing import Optional, Union, Tuple

import torch
from torch import Tensor

def _get_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: Optional[Union[str, int, Tuple[int]]] = None,
    dilation: int = 1,
    bias: bool = True,
    w_init_gain: str = "linear"
) -> torch.nn.Conv1d:
    
    if padding is None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        padding = int(dilation * (kernel_size - 1) / 2)

    conv1d = torch.nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    )

    torch.nn.init.xavier_uniform_(conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    return conv1d


def _get_linear_layer(in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear") -> torch.nn.Linear:

    linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
    torch.nn.init.xavier_uniform_(linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return linear


def _get_mask_from_lengths(lengths: Tensor) -> Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask
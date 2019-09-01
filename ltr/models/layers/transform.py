import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def interpolate(x, sz):
    """Interpolate 4D tensor x to size sz."""
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    return F.interpolate(x, sz, mode='bilinear', align_corners=False) if x.shape[-2:] != sz else x


class InterpCat(nn.Module):
    """Interpolate and concatenate features of different resolutions."""

    def forward(self, input):
        if isinstance(input, (dict, OrderedDict)):
            input = list(input.values())

        output_shape = None
        for x in input:
            if output_shape is None or output_shape[0] > x.shape[-2]:
                output_shape = x.shape[-2:]

        return torch.cat([interpolate(x, output_shape) for x in input], dim=-3)

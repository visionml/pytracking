import torch
import torch.nn as nn
import numpy as np

from spatial_correlation_sampler import SpatialCorrelationSampler


class CostVolume(nn.Module):
    def __init__(self, kernel_size, max_displacement, stride=1, abs_coordinate_output=False):
        super().__init__()
        self.correlation_layer = SpatialCorrelationSampler(kernel_size, 2*max_displacement + 1, stride,
                                                           int((kernel_size-1)/2))
        self.abs_coordinate_output = abs_coordinate_output

    def forward(self, feat1, feat2):
        assert feat1.dim() == 4 and feat2.dim() == 4, 'Expect 4 dimensional inputs'

        batch_size = feat1.shape[0]

        cost_volume = self.correlation_layer(feat1, feat2)

        if self.abs_coordinate_output:
            cost_volume = cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])
            cost_volume = remap_cost_volume(cost_volume)

        return cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])


def remap_cost_volume(cost_volume):
    """

    :param cost_volume: cost volume of shape (batch, (2*md-1)*(2*md-1), rows, cols), where md is the maximum displacement
                        allowed when computing the cost volume.
    :return: cost_volume_remapped: The input cost volume is remapped to shape (batch, rows, cols, rows, cols)
    """

    if cost_volume.dim() != 4:
        raise ValueError('input cost_volume should have 4 dimensions')

    [batch_size, d_, num_rows, num_cols] = cost_volume.size()
    d_sqrt_ = np.sqrt(d_)

    if not d_sqrt_.is_integer():
        raise ValueError("Invalid cost volume")

    cost_volume = cost_volume.view(batch_size, int(d_sqrt_), int(d_sqrt_), num_rows, num_cols)

    cost_volume_remapped = torch.zeros((batch_size, num_rows, num_cols,
                                        num_rows, num_cols),
                                       dtype=cost_volume.dtype,
                                       device=cost_volume.device)

    if cost_volume.size()[1] % 2 != 1:
        raise ValueError

    md = int((cost_volume.size()[1]-1)/2)

    for r in range(num_rows):
        for c in range(num_cols):
            r1_ = r - md
            r2_ = r1_ + 2*md + 1
            c1_ = c - md
            c2_ = c1_ + 2*md + 1

            r1_pad_ = max(-r1_, 0)
            r2_pad_ = max(r2_ - cost_volume_remapped.shape[1], 0)

            c1_pad_ = max(-c1_, 0)
            c2_pad_ = max(c2_ - cost_volume_remapped.shape[2], 0)

            d_ = cost_volume.size()[1]
            cost_volume_remapped[:, r1_+r1_pad_:r2_-r2_pad_, c1_+c1_pad_:c2_-c2_pad_, r, c] = \
                cost_volume[:, r1_pad_:d_-r2_pad_, c1_pad_:d_-c2_pad_, r, c]

    return cost_volume_remapped

import torch
import torch.nn as nn
import math
import ltr.models.layers.filter as filter_layer
from pytracking import TensorList


class LWTLResidual(nn.Module):
    """ Computes the residuals W(y_t)*(T_tau(x_t) - E(y_t) and lambda*tau in the few-shot learner loss (3) in the
    paper """
    def __init__(self, init_filter_reg=1e-2, filter_dilation_factors=None):
        super().__init__()
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.filter_dilation_factors = filter_dilation_factors

    def forward(self, meta_parameter: TensorList, feat, label, sample_weight=None):
        # Assumes multiple filters, i.e.  (sequences, filters, feat_dim, fH, fW)
        filter = meta_parameter[0]

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        # Compute scores
        scores = filter_layer.apply_filter(feat, filter, dilation_factors=self.filter_dilation_factors)

        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            if sample_weight.numel() == scores.numel():
                sample_weight = sample_weight.view(scores.shape)
            elif sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1, 1, 1)

        label = label.view(scores.shape)

        data_residual = sample_weight * (scores - label)

        # Compute regularization residual. Put batch in second dimension
        reg_residual = self.filter_reg*filter.view(1, num_sequences, -1)

        return TensorList([data_residual, reg_residual])

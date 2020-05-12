import torch
import torch.nn as nn
import math
import ltr.models.layers.filter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
from pytracking import TensorList


class LinearFilterLearnGen(nn.Module):
    def __init__(self, feat_stride=16, init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                 mask_init_factor=4.0, score_act='bentpar', act_param=None, mask_act='sigmoid'):
        super().__init__()

        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.feat_stride = feat_stride
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
        else:
            raise ValueError('Unknown activation')


    def forward(self, meta_parameter: TensorList, feat, bb, sample_weight=None, is_distractor=None):
        filter = meta_parameter[0]

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (filter.shape[-2], filter.shape[-1])

        # Compute scores
        scores = filter_layer.apply_filter(feat, filter)

        # Compute distance map
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1,))
        if is_distractor is not None:
            center[is_distractor.reshape(-1), :] = 99999
        dist_map = self.distance_map(center, scores.shape[-2:])

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])

        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(-1, 1, 1, 1) * spatial_weight

        # Compute data residual
        scores_act = self.score_activation(scores, target_mask)
        data_residual = sample_weight * (scores_act - label_map)

        # Compute regularization residual. Put batch in second dimension
        reg_residual = self.filter_reg*filter.reshape(1, num_sequences, -1)

        return TensorList([data_residual, reg_residual])

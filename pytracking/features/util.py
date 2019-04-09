import torch
from pytracking.features.featurebase import FeatureBase


class Concatenate(FeatureBase):
    """A feature that concatenates other features.
    args:
        features: List of features to concatenate.
    """
    def __init__(self, features, pool_stride = None, normalize_power = None, use_for_color = True, use_for_gray = True):
        super(Concatenate, self).__init__(pool_stride, normalize_power, use_for_color, use_for_gray)
        self.features = features

        self.input_stride = self.features[0].stride()

        for feat in self.features:
            if self.input_stride != feat.stride():
                raise ValueError('Strides for the features must be the same for a bultiresolution feature.')

    def dim(self):
        return sum([f.dim() for f in self.features])

    def stride(self):
        return self.pool_stride * self.input_stride

    def extract(self, im: torch.Tensor):
        return torch.cat([f.get_feature(im) for f in self.features], 1)

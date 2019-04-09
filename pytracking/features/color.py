import torch
from pytracking.features.featurebase import FeatureBase


class RGB(FeatureBase):
    """RGB feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 3

    def stride(self):
        return self.pool_stride

    def extract(self, im: torch.Tensor):
        return im/255 - 0.5


class Grayscale(FeatureBase):
    """Grayscale feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 1

    def stride(self):
        return self.pool_stride

    def extract(self, im: torch.Tensor):
        return torch.mean(im/255 - 0.5, 1, keepdim=True)

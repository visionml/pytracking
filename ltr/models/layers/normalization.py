import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


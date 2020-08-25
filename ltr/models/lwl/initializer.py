import torch.nn as nn


class FilterInitializerZero(nn.Module):
    """Initializes a target model with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, num_filters=1, feature_dim=256, filter_groups=1):
        super().__init__()

        self.filter_size = (num_filters, feature_dim//filter_groups, filter_size, filter_size)

    def forward(self, feat, mask=None):
        assert feat.dim() == 5
        # num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_sequences = feat.shape[1]

        return feat.new_zeros(num_sequences, *self.filter_size)

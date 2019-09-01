import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.layers.blocks import conv_block
import math


class FilterPool(nn.Module):
    """Pool the target region in a feature map.
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(filter_size, filter_size, 1/feature_stride)
        self.pool_square = pool_square

    def forward(self, feat, bb):
        """Pool the regions in bb.
        args:
            feat:  Input feature maps. Dims (num_samples, feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (num_samples, 4).
        returns:
            pooled_feat:  Pooled features. Dims (num_samples, feat_dim, wH, wW)."""

        # Add batch_index to rois
        bb = bb.view(-1,4)
        num_images_total = bb.shape[0]
        batch_index = torch.arange(num_images_total, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        pool_bb = bb.clone()

        if self.pool_square:
            bb_sz = pool_bb[:, 2:4].prod(dim=1, keepdim=True).sqrt()
            pool_bb[:, :2] += pool_bb[:, 2:]/2 - bb_sz/2
            pool_bb[:, 2:] = bb_sz

        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        roi1 = torch.cat((batch_index, pool_bb), dim=1)

        return self.prroi_pool(feat, roi1)



class FilterInitializer(nn.Module):
    """Initializes a target classification filter by applying a number of conv layers before and after pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        num_filter_pre_convs:  Conv layers before pooling.
        num_filter_post_convs:  Conv layers after pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16, pool_square=False, filter_norm=True,
                 num_filter_pre_convs=1, num_filter_post_convs=0):
        super().__init__()

        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm

        # Make pre conv
        pre_conv_layers = []
        for i in range(num_filter_pre_convs):
            pre_conv_layers.append(conv_block(feature_dim, feature_dim, kernel_size=3, padding=1))
        self.filter_pre_layers = nn.Sequential(*pre_conv_layers) if pre_conv_layers else None

        # Make post conv
        post_conv_layers = []
        for i in range(num_filter_post_convs):
            post_conv_layers.append(conv_block(feature_dim, feature_dim, kernel_size=1, padding=0))
        post_conv_layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0))
        self.filter_post_layers = nn.Sequential(*post_conv_layers)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""

        num_images = bb.shape[0] if bb.dim() == 3 else 1

        if self.filter_pre_layers is not None:
            feat = self.filter_pre_layers(feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))

        feat_post = self.filter_pool(feat, bb)
        weights = self.filter_post_layers(feat_post)

        if num_images > 1:
            weights = torch.mean(weights.view(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])

        return weights


class FilterInitializerLinear(nn.Module):
    """Initializes a target classification filter by applying a linear conv layer and then pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        conv_ksz:  Kernel size of the conv layer before pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16, pool_square=False, filter_norm=True,
                 conv_ksz=3):
        super().__init__()

        self.filter_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=conv_ksz, padding=conv_ksz // 2)
        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""

        num_images = feat.shape[0]

        feat = self.filter_conv(feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))

        weights = self.filter_pool(feat, bb)

        # If multiple input images, compute the initial filter as the average filter.
        if num_images > 1:
            weights = torch.mean(weights.view(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])

        return weights



class FilterInitializerZero(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, feature_dim=256):
        super().__init__()

        self.filter_size = (feature_dim, filter_size, filter_size)

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""

        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        return feat.new_zeros(num_sequences, self.filter_size[0], self.filter_size[1], self.filter_size[2])


class FilterInitializerSiamese(nn.Module):
    """Initializes a target classification filter by only pooling the target region (similar to Siamese trackers).
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False, filter_norm=True):
        super().__init__()

        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""

        num_images = feat.shape[0]

        feat = feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        weights = self.filter_pool(feat, bb)

        if num_images > 1:
            weights = torch.mean(weights.view(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])

        return weights

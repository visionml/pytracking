import torch
import torch.nn.functional as F
from pytracking import TensorList


class FeatureBase:
    """Base feature class.
    args:
        fparams: Feature specific parameters.
        pool_stride: Amount of average pooling to apply do downsample the feature map.
        output_size: Alternatively, specify the output size of the feature map. Adaptive average pooling will be applied.
        normalize_power: The power exponent for the normalization. None means no normalization (default).
        use_for_color: Use this feature for color images.
        use_for_gray: Use this feature for grayscale images.
    """
    def __init__(self, fparams = None, pool_stride = None, output_size = None, normalize_power = None, use_for_color = True, use_for_gray = True):
        self.fparams = fparams
        self.pool_stride = 1 if pool_stride is None else pool_stride
        self.output_size = output_size
        self.normalize_power = normalize_power
        self.use_for_color = use_for_color
        self.use_for_gray = use_for_gray

    def initialize(self):
        pass

    def dim(self):
        raise NotImplementedError

    def stride(self):
        raise NotImplementedError

    def size(self, im_sz):
        if self.output_size is None:
            return im_sz // self.stride()
        if isinstance(im_sz, torch.Tensor):
            return torch.Tensor([self.output_size[0], self.output_size[1]])
        return self.output_size

    def extract(self, im):
        """Performs feature extraction."""
        raise NotImplementedError

    def get_feature(self, im: torch.Tensor):
        """Get the feature. Generally, call this function.
        args:
            im: image patch as a torch.Tensor.
        """

        # Return empty tensor if it should not be used
        is_color = im.shape[1] == 3
        if is_color and not self.use_for_color or not is_color and not self.use_for_gray:
            return torch.Tensor([])

        # Extract feature
        feat = self.extract(im)

        # Pool/downsample
        if self.output_size is not None:
            feat = F.adaptive_avg_pool2d(feat, self.output_size)
        elif self.pool_stride != 1:
            feat = F.avg_pool2d(feat, self.pool_stride, self.pool_stride)

        # Normalize
        if self.normalize_power is not None:
            feat /= (torch.sum(feat.abs().view(feat.shape[0],1,1,-1)**self.normalize_power, dim=3, keepdim=True) /
                     (feat.shape[1]*feat.shape[2]*feat.shape[3]) + 1e-10)**(1/self.normalize_power)

        return feat


class MultiFeatureBase(FeatureBase):
    """Base class for features potentially having multiple feature blocks as output (like CNNs).
    See FeatureBase for more info.
    """
    def size(self, im_sz):
        if self.output_size is None:
            return TensorList([im_sz // s for s in self.stride()])
        if isinstance(im_sz, torch.Tensor):
            return TensorList([im_sz // s if sz is None else torch.Tensor([sz[0], sz[1]]) for sz, s in zip(self.output_size, self.stride())])

    def get_feature(self, im: torch.Tensor):
        """Get the feature. Generally, call this function.
        args:
            im: image patch as a torch.Tensor.
        """

        # Return empty tensor if it should not be used
        is_color = im.shape[1] == 3
        if is_color and not self.use_for_color or not is_color and not self.use_for_gray:
            return torch.Tensor([])

        feat_list = self.extract(im)

        output_sz = [None]*len(feat_list) if self.output_size is None else self.output_size

        # Pool/downsample
        for i, (sz, s) in enumerate(zip(output_sz, self.pool_stride)):
            if sz is not None:
                feat_list[i] = F.adaptive_avg_pool2d(feat_list[i], sz)
            elif s != 1:
                feat_list[i] = F.avg_pool2d(feat_list[i], s, s)

        # Normalize
        if self.normalize_power is not None:
            for feat in feat_list:
                feat /= (torch.sum(feat.abs().view(feat.shape[0],1,1,-1)**self.normalize_power, dim=3, keepdim=True) /
                         (feat.shape[1]*feat.shape[2]*feat.shape[3]) + 1e-10)**(1/self.normalize_power)

        return feat_list
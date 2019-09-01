import torch
from pytracking.utils.loading import load_network


class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter=0
    def __init__(self, net_path, use_gpu=True):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        self.net = load_network(self.net_path)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self):
        self.load_network()


class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""
    def initialize(self):
        super().initialize()
        self._mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self._std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        im = im/255
        im -= self._mean
        im /= self._std

        if self.use_gpu:
            im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        im = self.preprocess_image(im)
        return self.net.extract_backbone_features(im)

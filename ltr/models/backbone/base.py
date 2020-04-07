import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """
    def __init__(self, frozen_layers=()):
        super().__init__()

        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError('Unknown option for frozen layers: \"{}\". Should be \"all\", \"none\" or list of layer names.'.format(frozen_layers))

        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False


    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True


    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()


    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)
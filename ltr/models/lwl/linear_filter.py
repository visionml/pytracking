import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList


class LinearFilter(nn.Module):
    """ Target model constituting a single conv layer, along with the few-shot learner used to obtain the target model
        parameters (referred to as filter), i.e. weights of the conv layer
    """
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None,
                 filter_dilation_factors=None):
        super().__init__()

        self.filter_size = filter_size

        self.feature_extractor = feature_extractor          # Extracts features input to the target model

        self.filter_initializer = filter_initializer        # Predicts an initial filter in a feed-forward manner
        self.filter_optimizer = filter_optimizer            # Iteratively updates the filter by minimizing the few-shot
                                                            # learning loss

        self.filter_dilation_factors = filter_dilation_factors

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_label, *args, **kwargs):
        """ the mask should be 5d"""
        assert train_label.dim() == 5

        num_sequences = train_label.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract target model features
        train_feat = self.extract_target_model_features(train_feat, num_sequences)
        test_feat = self.extract_target_model_features(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat, train_label,
                                                 *args, **kwargs)

        # Predict mask encodings for the test frames
        mask_encodings = [self.apply_target_model(f, test_feat) for f in filter_iter]

        return mask_encodings

    def extract_target_model_features(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def apply_target_model(self, weights, feat):
        """ Apply the target model to obtain the mask encodings"""
        mask_encoding = filter_layer.apply_filter(feat, weights, dilation_factors=self.filter_dilation_factors)
        return mask_encoding

    def get_filter(self, feat, train_label, train_sw, num_objects=None, *args, **kwargs):
        """ Get the initial target model parameters given the few-shot labels """
        if num_objects is None:
            weights = self.filter_initializer(feat, train_label)
        else:
            weights = self.filter_initializer(feat, train_label)
            weights = weights.repeat(1, num_objects, 1, 1, 1)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(TensorList([weights]), feat=feat, label=train_label,
                                                                  sample_weight=train_sw,
                                                                  *args, **kwargs)
            weights = weights[0]
            weights_iter = [w[0] for w in weights_iter]
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

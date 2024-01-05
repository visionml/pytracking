import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict
import ltr.models.layers.filter as filter_layer


def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, c)).reshape(filter.shape)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        if attention.dim() == 4:
            feats_att = attention.unsqueeze(2) * feat  # (nf, ns, c, h, w)
        else:
            feats_att = feat.unsqueeze(2) * attention.unsqueeze(3)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0)  # (1, nf*ns, 4, h, w)

        if attention.dim() == 5:
            ltrb = ltrb.reshape(1, feats_att.shape[1], feats_att.shape[2], 4, feats_att.shape[4],
                                feats_att.shape[5])  # (1, nf*ns, num_obj, 4, h, w)

        return ltrb


class FPN(nn.Module):
    def __init__(self, high_res_layer='layer2', output_dim=256, input_dims=(512, 256)):
        super().__init__()
        self.high_res_layer = high_res_layer
        self.fpn = ops.FeaturePyramidNetwork(input_dims, output_dim)

    def forward(self, test_feat_enc, test_feat_backbone):
        num_sequences = test_feat_enc.shape[1]
        test_feat_enc = test_feat_enc.reshape(-1, *test_feat_enc.shape[-3:])

        d = OrderedDict()
        d['feat2'] = test_feat_backbone[self.high_res_layer]
        d['feat3'] = test_feat_enc
        output = self.fpn(d)

        for key in output.keys():
            output[key] = output[key].reshape(-1, num_sequences, *output[key].shape[-3:])

        return output


class FPNHead(nn.Module):
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor, fpn,
                 head_layer, separate_filters_for_cls_and_bbreg=False,
                 cls_output_mode=['high'], bbreg_output_mode=['high']):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.fpn = fpn
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg
        self.head_layer = head_layer
        self.cls_output_mode = cls_output_mode
        self.bbreg_output_mode = bbreg_output_mode

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        num_sequences = test_feat[self.head_layer].shape[0]

        for key in train_feat.keys():
            if train_feat[key].dim() == 5:
                train_feat[key] = train_feat[key].reshape(-1, *train_feat[key].shape[-3:])
        for key in test_feat.keys():
            if test_feat[key].dim() == 5:
                test_feat[key] = test_feat[key].reshape(-1, *test_feat[key].shape[-3:])

        # Extract features
        train_head_feat = self.extract_head_feat(train_feat, num_sequences)
        test_head_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        kwargs['train_bb'] = train_bb
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_head_feat, test_head_feat, *args,
                                                                              **kwargs)

        test_feat_enc_fpn = self.fpn(test_feat_enc, test_feat)

        # fuse encoder and decoder features to one feature map
        target_scores = OrderedDict()
        if 'trafo' in self.cls_output_mode:
            target_scores['trafo'] = self.classifier(test_feat_enc, cls_filter)
        if 'low' in self.cls_output_mode:
            target_scores['lowres'] = self.classifier(test_feat_enc_fpn['feat3'], cls_filter)
        if 'high' in self.cls_output_mode:
            target_scores['highres'] = self.classifier(test_feat_enc_fpn['feat2'], cls_filter)

        # compute the final prediction using the output module
        bbox_preds = OrderedDict()
        if 'trafo' in self.bbreg_output_mode:
            bbox_preds['trafo'] = self.bb_regressor(test_feat_enc, breg_filter)
        if 'low' in self.bbreg_output_mode:
            bbox_preds['lowres'] = self.bb_regressor(test_feat_enc_fpn['feat3'], breg_filter)
        if 'high' in self.bbreg_output_mode:
            bbox_preds['highres'] = self.bb_regressor(test_feat_enc_fpn['feat2'], breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat[self.head_layer]
        if num_sequences is None:
            return self.feature_extractor(feat[self.head_layer])

        output = self.feature_extractor(feat[self.head_layer])
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args,
                                                                              **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc

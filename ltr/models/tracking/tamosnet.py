import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads


class TaMOsNet(nn.Module):
    """The TaMOs network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tamosnet_resnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
                      final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                      num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, num_tokens=10, label_enc='gaussian', box_enc='ltrb',
                      fpn_head_cls_output_mode=['high'], fpn_head_bbreg_output_mode=['high'], **kwargs):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    elif isinstance(head_layer, list):
        if head_layer[-1] == 'layer3':
            feature_dim = 256
        elif head_layer[-1] == 'layer4':
            feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.GOTFilterPredictor(transformer, feature_sz=feature_sz, num_tokens=num_tokens,
                                                         label_enc=label_enc, box_enc=box_enc)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    fpn = heads.FPN()

    head = heads.FPNHead(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                    classifier=classifier, bb_regressor=bb_regressor, fpn=fpn, head_layer=head_layer[-1],
                    cls_output_mode=fpn_head_cls_output_mode, bbreg_output_mode=fpn_head_bbreg_output_mode)


    # ToMP network
    net = TaMOsNet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def tamosnet_swin_base(filter_size=4, head_layer='2', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
                    final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                    num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, num_tokens=10, label_enc='gaussian', box_enc='ltrb',
                    fpn_head_cls_output_mode=['high'], fpn_head_bbreg_output_mode=['high'], **kwargs):
    # Backbone
    backbone_net = backbones.swin_base384_flex(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == '2':
        feature_dim = 256
    elif head_layer == '3':
        feature_dim = 512
    elif isinstance(head_layer, list):
        if head_layer[-1] == '2':
            feature_dim = 256
        elif head_layer[-1] == '3':
            feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(input_dim=512, feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.GOTFilterPredictor(transformer, feature_sz=feature_sz, num_tokens=num_tokens,
                                             label_enc=label_enc, box_enc=box_enc)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    fpn = heads.FPN(high_res_layer='1', output_dim=256, input_dims=(256, 256))

    head = heads.FPNHead(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                         classifier=classifier, bb_regressor=bb_regressor, fpn=fpn, head_layer=head_layer[-1],
                         cls_output_mode=fpn_head_cls_output_mode, bbreg_output_mode=fpn_head_bbreg_output_mode)


    # TaMOs network
    net = TaMOsNet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net

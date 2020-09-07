import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.kys.predictor_wrapper as predictor_wrappers
import ltr.models.kys.response_predictor as resp_pred
import ltr.models.kys.cost_volume as cost_volume
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor


class KYSNet(nn.Module):
    def train(self, mode=True):
        self.training = mode

        self.backbone_feature_extractor.train(False)
        self.dimp_classifier.train(False)
        self.predictor.train(mode)
        self.bb_regressor.train(mode)

        if self.motion_feat_extractor is not None:
            self.motion_feat_extractor.train(mode)
        return self

    def __init__(self, backbone_feature_extractor, dimp_classifier, predictor,
                 bb_regressor, classification_layer, bb_regressor_layer, train_feature_extractor=True,
                 train_iounet=True, motion_feat_extractor=None, motion_layer=()):
        super().__init__()
        assert not train_feature_extractor
        self.backbone_feature_extractor = backbone_feature_extractor
        self.dimp_classifier = dimp_classifier
        self.predictor = predictor
        self.bb_regressor = bb_regressor
        self.classification_layer = classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.motion_layer = list(motion_layer)
        self.output_layers = sorted(list(set([self.classification_layer] + self.bb_regressor_layer + self.motion_layer)))
        self.train_iounet = train_iounet
        self.motion_feat_extractor = motion_feat_extractor

        if not train_feature_extractor:
            for p in self.backbone_feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, test_image_cur, dimp_filters, test_label_cur, backbone_feat_prev, label_prev,
                anno_prev, dimp_scores_prev, state_prev, dimp_jitter_fn):
        raise NotImplementedError

    def train_classifier(self, train_imgs, train_bb):
        assert train_imgs.dim() == 5, 'Expect 5 dimensions for train'

        num_sequences = train_imgs.shape[1]
        num_train_images = train_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Classification features
        train_feat_clf = train_feat[self.classification_layer]
        train_feat_clf = train_feat_clf.view(num_train_images, num_sequences, train_feat_clf.shape[-3],
                                             train_feat_clf.shape[-2], train_feat_clf.shape[-1])

        filter, train_losses = self.dimp_classifier.train_classifier(train_feat_clf, train_bb)
        return filter

    def extract_backbone_features(self, im, layers=None):
        im = im.view(-1, *im.shape[-3:])
        if layers is None:
            layers = self.output_layers

        return self.backbone_feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = backbone_feat[self.classification_layer]

        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.dimp_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def get_motion_feat(self, backbone_feat):
        if self.motion_feat_extractor is not None:
            motion_feat = self.motion_feat_extractor(backbone_feat)
            return motion_feat
        else:
            return self.predictor.extract_motion_feat(backbone_feat[self.classification_layer])

    def extract_features(self, im, layers):
        if 'classification' not in layers:
            return self.backbone_feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + [self.classification_layer] if l != 'classification' and l != 'motion'])))
        all_feat = self.backbone_feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.dimp_classifier.extract_classification_feat(all_feat[self.classification_layer])

        if self.motion_feat_extractor is not None:
            motion_feat = self.motion_feat_extractor(all_feat)
            all_feat['motion'] = motion_feat
        else:
            all_feat['motion'] = self.predictor.extract_motion_feat(all_feat[self.classification_layer])

        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def kysnet_res50(filter_size=4, optim_iter=3, appearance_feature_dim=512,
                 optim_init_step=0.9, optim_init_reg=0.1, classification_layer='layer3', backbone_pretrained=True,
                 clf_feat_blocks=0, clf_feat_norm=True, final_conv=True, init_filter_norm=False,
                 mask_init_factor=3.0, score_act='relu', target_mask_act='sigmoid', num_dist_bins=100,
                 bin_displacement=0.1, detach_length=float('Inf'),train_feature_extractor=True, train_iounet=True,
                 iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                 cv_kernel_size=3, cv_max_displacement=9, cv_stride=1,
                 init_gauss_sigma=1.0,
                 state_dim=8, representation_predictor_dims=(64, 32), gru_ksz=3,
                 conf_measure='max', dimp_thresh=None):

    # ######################## backbone ########################
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (appearance_feature_dim * filter_size * filter_size))

    # ######################## classifier ########################
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=appearance_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=appearance_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=16,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=None, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    cost_volume_layer = cost_volume.CostVolume(cv_kernel_size, cv_max_displacement, stride=cv_stride,
                                               abs_coordinate_output=True)

    motion_response_predictor = resp_pred.ResponsePredictor(state_dim=state_dim,
                                                            representation_predictor_dims=representation_predictor_dims,
                                                            gru_ksz=gru_ksz,
                                                            conf_measure=conf_measure,
                                                            dimp_thresh=dimp_thresh)

    response_predictor = predictor_wrappers.PredictorWrapper(cost_volume_layer, motion_response_predictor)

    net = KYSNet(backbone_feature_extractor=backbone_net, dimp_classifier=classifier,
                 predictor=response_predictor,
                 bb_regressor=bb_regressor,
                 classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'],
                 train_feature_extractor=train_feature_extractor,
                 train_iounet=train_iounet)
    return net

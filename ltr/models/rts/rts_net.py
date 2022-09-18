import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.rts.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.linear_filter as clf_target_clf
import ltr.models.target_classifier.initializer as clf_init
import ltr.models.target_classifier.residual_modules as clf_residual_modules

import ltr.models.rts.initializer as seg_initializer
import ltr.models.rts.label_encoder as seg_label_encoder
import ltr.models.rts.learners_fusion as fusion
import ltr.models.rts.loss_residual_modules as loss_residual_modules
import ltr.models.rts.decoder as rts_decoder
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.meta.steepestdescent as steepestdescent

from ltr.models.rts.utils import interpolate

from ltr import model_constructor
from pytracking import TensorList


class RTSNet(nn.Module):
    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers,
                 label_encoder=None, classifier=None, clf_encoder=None, classification_layer='layer3', clf_enc_input='baseline',
                 box_label_encoder=None, box_label_decoder=None, box_target_model=None,
                 box_target_model_segm=None, bbox_encoder=None, segm_encoder=None, fusion_module=None):
        super().__init__()

        # BBox to mask initialization
        self.box_target_model = box_target_model
        self.box_target_model_segm = box_target_model_segm
        self.bbox_encoder = bbox_encoder
        self.segm_encoder = segm_encoder
        self.box_label_encoder = box_label_encoder
        self.box_label_decoder = box_label_decoder

        # Segmentation learner
        self.target_model = target_model
        self.label_encoder = label_encoder

        self.target_model_input_layer = target_model_input_layer
        if isinstance(target_model_input_layer, str):
            self.target_model_input_layer = (target_model_input_layer,)

        # Instance learner - DiMP Classifier
        self.clf_encoder = clf_encoder
        self.clf_enc_input = clf_enc_input
        self.classifier = classifier

        self.classification_layer = classification_layer
        if isinstance(classification_layer, str):
            self.classification_layer = (classification_layer,)

        # Common parts
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.fusion_module = fusion_module
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))


    def forward_box_mask_sta(self, train_imgs, train_bb):

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]

        # Extract backbone features
        train_feat_backbone = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Extract classification features
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)

        train_bbox_enc, _ = self.box_label_encoder(train_bb, train_feat_tm, list(train_imgs.shape[-2:]))
        train_mask_enc, train_mask_sw = self.bbox_encoder(train_bb, train_feat_tm, list(train_imgs.shape[-2:]))
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])

        _, filter_iter, _ = self.box_target_model.get_filter(train_feat_tm, train_mask_enc, train_mask_sw)
        target_scores = [self.box_target_model.apply_target_model(f, train_feat_tm) for f in filter_iter]
        target_scores_last_iter = target_scores[-1]
        coarse_mask = torch.cat((train_bb, target_scores_last_iter), dim=2)
        pred_all, _ = self.box_label_decoder(coarse_mask, train_feat_backbone, train_imgs.shape[-2:])

        pred_all = pred_all.view(num_train_frames, num_sequences, *pred_all.shape[-2:])
        train_segm_enc, train_segm_sw = self.segm_encoder(torch.sigmoid(pred_all), train_feat_tm)
        _, filter_iter_segm, _ = self.box_target_model_segm.get_filter(train_feat_tm, train_segm_enc, train_segm_sw)
        target_scores_segm = [self.box_target_model_segm.apply_target_model(f, train_feat_tm) for f in filter_iter_segm]
        target_scores_last_iter_segm = target_scores_segm[-1]
        coarse_mask = torch.cat((train_bb, target_scores_last_iter_segm), dim=2)
        pred_all_segm, _ = self.box_label_decoder(coarse_mask, train_feat_backbone, train_imgs.shape[-2:])
        pred_all_segm = pred_all_segm.view(num_train_frames, num_sequences, *pred_all_segm.shape[-2:])

        return pred_all, pred_all_segm

    def forward_classifier_only(self, train_imgs, test_imgs, train_bb, train_label):

        # Extract backbone features
        train_feat_backbone = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        train_feat_clf = self.get_backbone_clf_feat(train_feat_backbone)
        test_feat_clf = self.get_backbone_clf_feat(test_feat_backbone)

        clf_target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label=train_label)

        return clf_target_scores

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, train_bb, train_label, test_label, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat_backbone = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(
            test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract features input to the target model
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)
        test_feat_tm = self.extract_target_model_features(test_feat_backbone)

        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])
        test_feat_tm = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])

        train_feat_tm_all = [train_feat_tm, ]

        train_feat_clf = self.get_backbone_clf_feat(train_feat_backbone)
        test_feat_clf = self.get_backbone_clf_feat(test_feat_backbone)
        clf_target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label=train_label)

        # Get few-shot learner label and spatial importance weights

        few_shot_label, few_shot_sw = self.label_encoder(train_masks, train_feat_tm)
        few_shot_label_all = [few_shot_label, ]
        few_shot_sw_all = None if few_shot_sw is None else [few_shot_sw, ]

        if self.clf_enc_input in ['baseline', 'gt']:
            clf_input = test_label
        elif self.clf_enc_input == 'sc':
            clf_input = clf_target_scores[-1]
        else:
            print("unknown clf enc input mode")
            assert False
        encoded_bbox_labels, _ = self.clf_encoder(clf_input)

        # Obtain the target module parameters using the few-shot learner
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_tm, few_shot_label, few_shot_sw)

        mask_predictons_all = []

        # Iterate over the test sequence
        for i in range(num_test_frames):
            # Features for the current frame
            test_feat_tm_it = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])[i:i+1, ...]

            # Apply the target model to obtain mask encodings.
            mask_encoding_pred = [self.target_model.apply_target_model(f, test_feat_tm_it) for f in filter_iter]

            test_feat_backbone_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in
                                     test_feat_backbone.items()}
            mask_encoding_pred_last_iter = mask_encoding_pred[-1]

            # Run decoder to obtain the segmentation mask
            if self.clf_enc_input == 'baseline':
                decoder_input = mask_encoding_pred_last_iter
            else:
                encoded_bbox_label = interpolate(encoded_bbox_labels[i,:,:,:,:], mask_encoding_pred_last_iter.shape[-2:])
                encoded_bbox_label = encoded_bbox_label.unsqueeze(0)
                decoder_input = self.fusion_module(mask_encoding_pred_last_iter, encoded_bbox_label)

            mask_pred, decoder_feat = self.decoder(decoder_input, test_feat_backbone_it,
                                                   test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])

            mask_predictons_all.append(mask_pred)

            # Convert the segmentation scores to probability
            mask_pred_prob = torch.sigmoid(mask_pred.clone().detach())

            # Obtain label encoding for the predicted mask in the previous frame
            few_shot_label, few_shot_sw = self.label_encoder(mask_pred_prob, test_feat_tm_it)

            # Extend the training data using the predicted mask
            few_shot_label_all.append(few_shot_label)
            if few_shot_sw_all is not None:
                few_shot_sw_all.append(few_shot_sw)

            train_feat_tm_all.append(test_feat_tm_it)

            # Update the target model using the extended training set
            if (i < (num_test_frames - 1)) and (num_refinement_iter > 0):
                train_feat_tm_it = torch.cat(train_feat_tm_all, dim=0)
                few_shot_label_it = torch.cat(few_shot_label_all, dim=0)

                if few_shot_sw_all is not None:
                    few_shot_sw_it = torch.cat(few_shot_sw_all, dim=0)
                else:
                    few_shot_sw_it = None

                # Run few-shot learner to update the target model
                filter_updated, _, _ = self.target_model.filter_optimizer(TensorList([filter]),
                                                                          feat=train_feat_tm_it,
                                                                          label=few_shot_label_it,
                                                                          sample_weight=few_shot_sw_it,
                                                                          num_iter=num_refinement_iter)

                filter = filter_updated[0]      # filter_updated is a TensorList

        mask_predictons_all = torch.cat(mask_predictons_all, dim=0)
        return mask_predictons_all, clf_target_scores

    def segment_target(self, target_filter, test_feat_tm, test_feat, encoded_clf_scores=None):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])

        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)

        decoder_input = mask_encoding_pred
        if encoded_clf_scores is not None:
            encoded_clf_scores = interpolate(encoded_clf_scores[0,:,:,:,:], mask_encoding_pred.shape[-2:])
            encoded_clf_scores = encoded_clf_scores.unsqueeze(0)
            decoder_input = self.fusion_module(mask_encoding_pred, encoded_clf_scores)


        mask_pred, decoder_feat = self.decoder(decoder_input, test_feat,
                                               (test_feat_tm.shape[-2]*16, test_feat_tm.shape[-1]*16))

        return mask_pred, mask_encoding_pred

    def get_backbone_target_model_features(self, backbone_feat):
        # Get the backbone feature block which is input to the target model
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))


def build_base_components(filter_size, num_filters, optim_iter, optim_init_reg, backbone_pretrained,
                          clf_feat_blocks, clf_feat_norm, final_conv,
                          out_feature_dim, target_model_input_layer,
                          classification_layer, decoder_input_layers, detach_length,
                          label_encoder_dims, frozen_backbone_layers, decoder_mdim, filter_groups,
                          use_bn_in_label_enc, dilation_factors, backbone_type,
                          clf_with_extractor, clf_hinge_threshold, clf_feat_stride,
                          clf_activation_leak, clf_act_param, clf_score_act,
                          clf_filter_size):

    # backbone feature extractor F
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
    else:
        raise Exception

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        clf_feature_dim = 256
    elif classification_layer == 'layer4':
        clf_feature_dim = 512
    else:
        raise Exception

    layer_channels = backbone_net.out_feature_channels()

    # Extracts features input to the target model
    target_model_feature_extractor = clf_features.residual_basic_block(
        feature_dim=layer_channels[target_model_input_layer],
        num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
        final_conv=final_conv, norm_scale=norm_scale,
        out_dim=out_feature_dim)

    # Few-shot label generator and weight predictor
    label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters,),
                                                     use_bn=use_bn_in_label_enc)

    # Predicts initial target model parameters
    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    # Computes few-shot learning loss
    residual_module = loss_residual_modules.RTSResidual(init_filter_reg=optim_init_reg,
                                                         filter_dilation_factors=dilation_factors)

    # Iteratively updates the target model parameters by minimizing the few-shot learning loss
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    # Target model and Few-shot learner
    target_model = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                           filter_optimizer=optimizer, feature_extractor=target_model_feature_extractor,
                                           filter_dilation_factors=dilation_factors)

    # DiMP-like classifier

    # Residual module that defined the online loss
    clf_residual_module = clf_residual_modules.LinearFilterHinge(
        feat_stride=clf_feat_stride, init_filter_reg=optim_init_reg,
        hinge_threshold=clf_hinge_threshold, activation_leak=clf_activation_leak,
        score_act=clf_score_act, act_param=clf_act_param, learn_filter_reg=False)

    # Construct generic optimizer module
    clf_optimizer = steepestdescent.GNSteepestDescent(
        residual_module=clf_residual_module, num_iter=optim_iter, detach_length=detach_length,
        residual_batch_dim=1, compute_losses=True)

    if clf_with_extractor:
        clf_initializer = clf_init.FilterInitializerLinear(
            filter_size=clf_filter_size, filter_norm=False,
            feature_dim=out_feature_dim)
        clf_feature_extractor = clf_features.residual_bottleneck(
            feature_dim=clf_feature_dim,
            num_blocks=0, l2norm=True,
            final_conv=True, norm_scale=norm_scale,
            out_dim=out_feature_dim,
            final_stride=2)
    else:
        clf_initializer = clf_init.FilterInitializerZero(
            filter_size=clf_filter_size, feature_dim=out_feature_dim)
        clf_feature_extractor = None

    classifier = clf_target_clf.LinearFilter(
        filter_size=clf_filter_size, filter_initializer=clf_initializer,
        filter_optimizer=clf_optimizer, feature_extractor=clf_feature_extractor)


    # Decoder
    decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

    decoder = rts_decoder.RTSDecoder(num_filters, decoder_mdim, decoder_input_layers_channels, use_bn=True)

    return backbone_net, target_model, decoder, label_encoder, classifier, \
           layer_channels, decoder_input_layers_channels, norm_scale


@model_constructor
def steepest_descent_resnet50(
        filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
        backbone_pretrained=False, clf_feat_blocks=1,
        clf_feat_norm=True, final_conv=False,
        out_feature_dim=512,
        target_model_input_layer='layer3',
        classification_layer='layer3',
        decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
        detach_length=float('Inf'),
        label_encoder_dims=(1, 1),
        frozen_backbone_layers=(),
        decoder_mdim=64, filter_groups=1,
        use_bn_in_label_enc=True,
        dilation_factors=None,
        backbone_type='imagenet',
        clf_with_extractor=False,
        clf_hinge_threshold=0.05,
        clf_feat_stride=16,
        clf_activation_leak=0.1,
        clf_act_param=None,
        clf_score_act='relu',
        clf_filter_size=3):

    backbone_net, target_model, decoder, label_encoder, classifier, \
    layer_channels, decoder_input_layers_channels, norm_scale = build_base_components(
        filter_size=filter_size, num_filters=num_filters, optim_iter=optim_iter,
        optim_init_reg=optim_init_reg, backbone_pretrained=backbone_pretrained,
        clf_feat_blocks=clf_feat_blocks, clf_feat_norm=clf_feat_norm,
        final_conv=final_conv, out_feature_dim=out_feature_dim,
        target_model_input_layer=target_model_input_layer,
        classification_layer=classification_layer,
        decoder_input_layers=decoder_input_layers,
        detach_length=detach_length, label_encoder_dims=label_encoder_dims,
        frozen_backbone_layers=frozen_backbone_layers, decoder_mdim=decoder_mdim,
        filter_groups=filter_groups, use_bn_in_label_enc=use_bn_in_label_enc,
        dilation_factors=dilation_factors, backbone_type=backbone_type,
        clf_with_extractor=clf_with_extractor,
        clf_hinge_threshold=clf_hinge_threshold,
        clf_feat_stride=clf_feat_stride,
        clf_activation_leak=clf_activation_leak,
        clf_act_param=clf_act_param,
        clf_score_act=clf_score_act,
        clf_filter_size=clf_filter_size)

    net = RTSNet(
        feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
        label_encoder=label_encoder, classifier=classifier,
        target_model_input_layer=target_model_input_layer,
        decoder_input_layers=decoder_input_layers)

    return net

@model_constructor
def steepest_descent_resnet50_with_clf_encoder(
        filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
        backbone_pretrained=False, clf_feat_blocks=1,
        clf_feat_norm=True, final_conv=False,
        out_feature_dim=512,
        target_model_input_layer='layer3',
        classification_layer='layer3',
        decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
        detach_length=float('Inf'),
        label_encoder_dims=(1, 1),
        frozen_backbone_layers=(),
        decoder_mdim=64, filter_groups=1,
        use_bn_in_label_enc=True,
        dilation_factors=None,
        backbone_type='imagenet',
        clf_with_extractor=False,
        clf_hinge_threshold=0.05,
        clf_feat_stride=16,
        clf_activation_leak=0.1,
        clf_act_param=None,
        clf_score_act='relu',
        clf_filter_size=3,
        clf_enc_input='baseline',
        fusion_type='add'):

    backbone_net, target_model, decoder, label_encoder, classifier, \
    layer_channels, decoder_input_layers_channels, norm_scale = build_base_components(
        filter_size=filter_size, num_filters=num_filters, optim_iter=optim_iter,
        optim_init_reg=optim_init_reg, backbone_pretrained=backbone_pretrained,
        clf_feat_blocks=clf_feat_blocks, clf_feat_norm=clf_feat_norm,
        final_conv=final_conv, out_feature_dim=out_feature_dim,
        target_model_input_layer=target_model_input_layer,
        classification_layer=classification_layer,
        decoder_input_layers=decoder_input_layers,
        detach_length=detach_length, label_encoder_dims=label_encoder_dims,
        frozen_backbone_layers=frozen_backbone_layers, decoder_mdim=decoder_mdim,
        filter_groups=filter_groups, use_bn_in_label_enc=use_bn_in_label_enc,
        dilation_factors=dilation_factors, backbone_type=backbone_type,
        clf_with_extractor=clf_with_extractor,
        clf_hinge_threshold=clf_hinge_threshold,
        clf_feat_stride=clf_feat_stride,
        clf_activation_leak=clf_activation_leak,
        clf_act_param=clf_act_param,
        clf_score_act=clf_score_act,
        clf_filter_size=clf_filter_size)

    clf_encoder = seg_label_encoder.ResidualDS16SW_Clf(
        layer_dims=label_encoder_dims + (num_filters,),
        use_bn=use_bn_in_label_enc)

    fusion_module = fusion.LearnersFusion(fusion_type)

    net = RTSNet(
        feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
        label_encoder=label_encoder, classifier=classifier, clf_encoder=clf_encoder,
        target_model_input_layer=target_model_input_layer,
        decoder_input_layers=decoder_input_layers, clf_enc_input=clf_enc_input,
        fusion_module=fusion_module)

    return net


@model_constructor
def steepest_descent_resnet50_with_clf_encoder_boxinit(
        filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
        backbone_pretrained=False, clf_feat_blocks=1,
        clf_feat_norm=True, final_conv=False,
        out_feature_dim=512,
        target_model_input_layer='layer3',
        classification_layer='layer3',
        decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
        detach_length=float('Inf'),
        label_encoder_dims=(1, 1),
        frozen_backbone_layers=(),
        decoder_mdim=64, filter_groups=1,
        use_bn_in_label_enc=True,
        dilation_factors=None,
        backbone_type='imagenet',
        clf_with_extractor=False,
        clf_hinge_threshold=0.05,
        clf_feat_stride=16,
        clf_activation_leak=0.1,
        clf_act_param=None,
        clf_score_act='relu',
        clf_filter_size=3,
        clf_enc_input='baseline',
        fusion_type='add',
        box_label_encoder_type='ResidualDS16FeatSWBox'):

    backbone_net, target_model, decoder, label_encoder, classifier, \
    layer_channels, decoder_input_layers_channels, norm_scale = build_base_components(
        filter_size=filter_size, num_filters=num_filters, optim_iter=optim_iter,
        optim_init_reg=optim_init_reg, backbone_pretrained=backbone_pretrained,
        clf_feat_blocks=clf_feat_blocks, clf_feat_norm=clf_feat_norm,
        final_conv=final_conv, out_feature_dim=out_feature_dim,
        target_model_input_layer=target_model_input_layer,
        classification_layer=classification_layer,
        decoder_input_layers=decoder_input_layers,
        detach_length=detach_length, label_encoder_dims=label_encoder_dims,
        frozen_backbone_layers=frozen_backbone_layers, decoder_mdim=decoder_mdim,
        filter_groups=filter_groups, use_bn_in_label_enc=use_bn_in_label_enc,
        dilation_factors=dilation_factors, backbone_type=backbone_type,
        clf_with_extractor=clf_with_extractor,
        clf_hinge_threshold=clf_hinge_threshold,
        clf_feat_stride=clf_feat_stride,
        clf_activation_leak=clf_activation_leak,
        clf_act_param=clf_act_param,
        clf_score_act=clf_score_act,
        clf_filter_size=clf_filter_size)

    clf_encoder = seg_label_encoder.ResidualDS16SW_Clf(
        layer_dims=label_encoder_dims + (num_filters,),
        use_bn=use_bn_in_label_enc)

    fusion_module = fusion.LearnersFusion(fusion_type)

    # BOX INIT PART
    if box_label_encoder_type == 'ResidualDS16FeatSWBox':
        box_initializer = seg_initializer.FilterInitializerZero(
            filter_size=filter_size, num_filters=num_filters,
            feature_dim=out_feature_dim, filter_groups=filter_groups)
        box_initializer_segm = seg_initializer.FilterInitializerZero(
            filter_size=filter_size, num_filters=num_filters,
            feature_dim=out_feature_dim, filter_groups=filter_groups)

        box_label_encoder_dims = (16, 32, 64, 128)

        # Few-shot label generator and weight predictor
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBox(
            layer_dims=box_label_encoder_dims + (num_filters, ),
            feat_dim=out_feature_dim, use_final_relu=True,
            use_gauss=False)
        bbox_encoder = seg_label_encoder.ResidualDS16FeatSWBox(
            layer_dims=box_label_encoder_dims + (num_filters, ),
            feat_dim=out_feature_dim, use_final_relu=True,
            use_gauss=False)
        segm_encoder = seg_label_encoder.ResidualDS16SW(
            layer_dims=box_label_encoder_dims[:-1] + (num_filters, ),
            use_bn=use_bn_in_label_enc)

        # Computes few-shot learning loss
        box_residual_module = loss_residual_modules.RTSResidual(
            init_filter_reg=optim_init_reg)
        box_residual_module_segm = loss_residual_modules.RTSResidual(
            init_filter_reg=optim_init_reg)

        # Iteratively updates the target model params by minimizing the few-shot learning loss
        box_optimizer = steepestdescent.GNSteepestDescent(
            residual_module=box_residual_module, num_iter=optim_iter,
            detach_length=detach_length, residual_batch_dim=1, compute_losses=True)
        box_optimizer_segm = steepestdescent.GNSteepestDescent(
            residual_module=box_residual_module_segm, num_iter=optim_iter,
            detach_length=detach_length, residual_batch_dim=1, compute_losses=True)

        # Target model and Few-shot learner
        # Extracts features input to the target model
        box_target_model_feature_extractor = clf_features.residual_basic_block(
            feature_dim=layer_channels[target_model_input_layer],
            num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
            final_conv=final_conv, norm_scale=norm_scale,
            out_dim=out_feature_dim)

        box_target_model = target_clf.LinearFilter(
            filter_size=filter_size, filter_initializer=box_initializer,
            filter_optimizer=box_optimizer, feature_extractor=box_target_model_feature_extractor,
            filter_dilation_factors=dilation_factors)
        box_target_model_segm = target_clf.LinearFilter(
            filter_size=filter_size, filter_initializer=box_initializer_segm,
            filter_optimizer=box_optimizer_segm, feature_extractor=None,
            filter_dilation_factors=dilation_factors)

        # Decoder
        decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}
        box_label_decoder = rts_decoder.RTSDecoder(
            num_filters*2, decoder_mdim, decoder_input_layers_channels, use_bn=True)

    else:
        raise Exception

    net = RTSNet(
        feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
        label_encoder=label_encoder, classifier=classifier, clf_encoder=clf_encoder,
        target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers,
        clf_enc_input=clf_enc_input, box_label_encoder=box_label_encoder, box_label_decoder=box_label_decoder,
        box_target_model=box_target_model, box_target_model_segm=box_target_model_segm,
        bbox_encoder=bbox_encoder, segm_encoder=segm_encoder, fusion_module=fusion_module)

    return net

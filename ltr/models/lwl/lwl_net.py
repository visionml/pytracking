import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.lwl.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.lwl.initializer as seg_initializer
import ltr.models.lwl.label_encoder as seg_label_encoder
import ltr.models.lwl.loss_residual_modules as loss_residual_modules
import ltr.models.lwl.decoder as lwtl_decoder
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.meta.steepestdescent as steepestdescent
from ltr import model_constructor
from pytracking import TensorList


class LWTLNet(nn.Module):
    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers,
                 label_encoder=None):
        super().__init__()

        self.feature_extractor = feature_extractor      # Backbone feature extractor F
        self.target_model = target_model                    # Target model and the few-shot learner
        self.decoder = decoder                          # Segmentation Decoder

        self.label_encoder = label_encoder              # Few-shot label generator and weight predictor

        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer,
                                                                                  str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat_backbone = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract features input to the target model
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)  # seq*frames, channels, height, width
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])

        train_feat_tm_all = [train_feat_tm, ]

        # Get few-shot learner label and spatial importance weights
        few_shot_label, few_shot_sw = self.label_encoder(train_masks, train_feat_tm)

        few_shot_label_all = [few_shot_label, ]
        few_shot_sw_all = None if few_shot_sw is None else [few_shot_sw, ]

        test_feat_tm = self.extract_target_model_features(test_feat_backbone)  # seq*frames, channels, height, width

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
            mask_pred, decoder_feat = self.decoder(mask_encoding_pred_last_iter, test_feat_backbone_it,
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
        return mask_predictons_all

    def segment_target(self, target_filter, test_feat_tm, test_feat):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])

        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)

        mask_pred, decoder_feat = self.decoder(mask_encoding_pred, test_feat,
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


@model_constructor
def steepest_descent_resnet50(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=512,
                              target_model_input_layer='layer3',
                              decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              decoder_mdim=64, filter_groups=1,
                              use_bn_in_label_enc=True,
                              dilation_factors=None,
                              backbone_type='imagenet'):
    # backbone feature extractor F
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
    else:
        raise Exception

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

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
    residual_module = loss_residual_modules.LWTLResidual(init_filter_reg=optim_init_reg,
                                                         filter_dilation_factors=dilation_factors)

    # Iteratively updates the target model parameters by minimizing the few-shot learning loss
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    # Target model and Few-shot learner
    target_model = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                           filter_optimizer=optimizer, feature_extractor=target_model_feature_extractor,
                                           filter_dilation_factors=dilation_factors)

    # Decoder
    decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

    decoder = lwtl_decoder.LWTLDecoder(num_filters, decoder_mdim, decoder_input_layers_channels, use_bn=True)

    net = LWTLNet(feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
                  label_encoder=label_encoder,
                  target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers)
    return net

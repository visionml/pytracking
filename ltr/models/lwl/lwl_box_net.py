import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.lwl.label_encoder as seg_label_encoder
from ltr import model_constructor
import ltr.models.lwl.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.lwl.initializer as seg_initializer
import ltr.models.lwl.loss_residual_modules as loss_residual_modules
import ltr.models.lwl.decoder as lwtl_decoder
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.meta.steepestdescent as steepestdescent


class LWTLBoxNet(nn.Module):
    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers,
                 label_encoder=None, box_label_encoder=None):
        super().__init__()

        self.feature_extractor = feature_extractor      # Backbone feature extractor F
        self.target_model = target_model                    # Target model and the few-shot learner
        self.decoder = decoder                          # Segmentation Decoder

        self.label_encoder = label_encoder              # Few-shot label generator and weight predictor

        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer,
                                                                                  str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))
        self.box_label_encoder = box_label_encoder
        self.train_only_box_label_gen = True

    def train(self, mode=True):
        for x in self.feature_extractor.parameters():
            x.requires_grad_(False)
        self.feature_extractor.eval()

        if mode:
            for x in self.box_label_encoder.parameters():
                x.requires_grad_(True)
            self.box_label_encoder.train()

            if self.train_only_box_label_gen:
                for x in self.target_model.parameters():
                    x.requires_grad_(False)
                self.target_model.eval()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(False)
                self.label_encoder.eval()
                for x in self.decoder.parameters():
                    x.requires_grad_(False)
                self.decoder.eval()

            else:
                for x in self.target_model.parameters():
                    x.requires_grad_(True)
                self.target_model.train()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(True)
                self.label_encoder.train()
                for x in self.decoder.parameters():
                    x.requires_grad_(True)
                self.decoder.train()

        else:
            for x in self.target_model.parameters():
                x.requires_grad_(False)
            self.target_model.eval()

            for x in self.label_encoder.parameters():
                x.requires_grad_(False)
            self.label_encoder.eval()

            for x in self.decoder.parameters():
                x.requires_grad_(False)
            self.decoder.eval()

            for x in self.box_label_encoder.parameters():
                x.requires_grad_(False)
            self.box_label_encoder.eval()

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, bb_train, num_refinement_iter=2):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.contiguous().view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_classification_feat(train_feat)       # seq*frames, channels, height, width
        test_feat_clf = self.extract_classification_feat(test_feat)         # seq*frames, channels, height, width

        bb_mask_enc = self.box_label_encoder(bb_train, train_feat_clf)

        box_mask_pred, decoder_feat = self.decoder(bb_mask_enc, test_feat, test_imgs.shape[-2:],
                                               ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

        mask_enc = self.label_encoder(box_mask_pred, train_feat_clf)
        mask_enc_test = self.label_encoder(test_masks.contiguous(), test_feat_clf)

        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_clf, *mask_enc)

        test_feat_clf = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])
        target_scores = [self.target_model.classify(f, test_feat_clf) for f in filter_iter]
        # target_scores = [s.unsqueeze(dim=2) for s in target_scores]

        target_scores_last_iter = target_scores[-1]

        mask_pred, decoder_feat = self.decoder(target_scores_last_iter, test_feat, test_imgs.shape[-2:],
                                               ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

        decoder_feat['mask_enc'] = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])

        if isinstance(mask_enc_test, (tuple, list)):
            mask_enc_test = mask_enc_test[0]
        return mask_pred, target_scores, mask_enc_test, box_mask_pred

    def segment_target(self, target_filter, test_feat_tm, test_feat):
        # Classification features
        assert target_filter.dim() == 5  # seq, filters, ch, h, w
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])

        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)

        mask_pred, decoder_feat = self.decoder(mask_encoding_pred, test_feat,
                                               (test_feat_tm.shape[-2] * 16, test_feat_tm.shape[-1] * 16))

        return mask_pred, None

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
                              backbone_type='imagenet',
                              box_label_encoder_dims=(1, 1),
                              box_label_encoder_type='ResidualDS16FeatSWBoxCatMultiBlock',
                              use_gauss=False,
                              use_final_relu=True,
                              init_bn=1.0,
                              final_bn=False,
                              gauss_scale=0.25
                              ):
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
    residual_module = loss_residual_modules.LWTLResidual(init_filter_reg=optim_init_reg)

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

    if box_label_encoder_type == 'ResidualDS16FeatSWBoxCatMultiBlock':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxCatMultiBlock(feat_dim=out_feature_dim,
                                                                                 layer_dims=box_label_encoder_dims + (
                                                                                     num_filters,),
                                                                                 use_gauss=use_gauss,
                                                                                 use_final_relu=use_final_relu,
                                                                                 use_bn=use_bn_in_label_enc,
                                                                                 non_default_init=True, init_bn=init_bn,
                                                                                 gauss_scale=gauss_scale,
                                                                                 final_bn=final_bn)

    else:
        raise Exception

    net = LWTLBoxNet(feature_extractor=backbone_net, target_model=target_model, decoder=decoder,
                     label_encoder=label_encoder,
                     target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers,
                     box_label_encoder=box_label_encoder)
    return net


@model_constructor
def steepest_descent_resnet50_from_checkpoint(net=None, num_filters=1,
                                              out_feature_dim=512,
                                              target_model_input_layer='layer3',
                                              box_label_encoder_dims=(1, 1),
                                              use_bn_in_label_enc=True,
                                              box_label_encoder_type='ResidualDS16FeatSWBoxCatMultiBlock',
                                              use_gauss=False,
                                              use_final_relu=True,
                                              init_bn=1,
                                              gauss_scale=0.25,
                                              decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
                                              final_bn=False):

    if box_label_encoder_type == 'ResidualDS16FeatSWBoxCatMultiBlock':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxCatMultiBlock(feat_dim=out_feature_dim,
                                                                                 layer_dims=box_label_encoder_dims + (
                                                                                 num_filters,),
                                                                                 use_gauss=use_gauss,
                                                                                 use_final_relu=use_final_relu,
                                                                                 use_bn=use_bn_in_label_enc,
                                                                                 non_default_init=True, init_bn=init_bn,
                                                                                 gauss_scale=gauss_scale,
                                                                                 final_bn=final_bn)

    else:
        raise Exception

    net = LWTLBoxNet(feature_extractor=net.feature_extractor, target_model=net.target_model, decoder=net.decoder,
                     label_encoder=net.label_encoder, target_model_input_layer=target_model_input_layer,
                     decoder_input_layers=decoder_input_layers,
                     box_label_encoder=box_label_encoder)
    return net


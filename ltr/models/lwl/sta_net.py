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


class STANet(nn.Module):
    def __init__(self, feature_extractor, target_model, target_model_segm, decoder, target_model_input_layer,
                 decoder_input_layers, label_encoder=None, bbox_encoder=None, segm_encoder=None):
        super().__init__()

        self.feature_extractor = feature_extractor          # Backbone feature extractor F
        self.target_model = target_model                    # Target model for initial prediction
        self.target_model_segm = target_model_segm          # Target model for final prediction
        self.decoder = decoder                              # Segmentation Decoder

        self.label_encoder = label_encoder                  # Few-shot label generator and weight predictor
        self.bbox_encoder = bbox_encoder                    # Few-shot label generator and weight predictor
        self.segm_encoder = segm_encoder                    # Few-shot label generator and weight predictor

        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer,
                                                                                  str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward(self, train_imgs, train_bbox):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(
            train_imgs.reshape(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_target_model_features(train_feat)  # seq*frames, channels, height, width

        train_bbox_enc, _ = self.label_encoder(train_bbox, train_feat_clf, list(train_imgs.shape[-2:]))
        train_mask_enc, train_mask_sw = self.bbox_encoder(train_bbox, train_feat_clf, list(train_imgs.shape[-2:]))
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        _, filter_iter, _ = self.target_model.get_filter(train_feat_clf, train_mask_enc, train_mask_sw)
        target_scores = [self.target_model.apply_target_model(f, train_feat_clf) for f in filter_iter]
        target_scores_last_iter = target_scores[-1]
        coarse_mask = torch.cat((train_bbox_enc, target_scores_last_iter), dim=2)
        pred_all, _ = self.decoder(coarse_mask, train_feat, train_imgs.shape[-2:])
        
        pred_all = pred_all.view(num_train_frames, num_sequences, *pred_all.shape[-2:])
        train_segm_enc, train_segm_sw = self.segm_encoder(torch.sigmoid(pred_all), train_feat_clf)
        _, filter_iter_segm, _ = self.target_model_segm.get_filter(train_feat_clf, train_segm_enc, train_segm_sw)
        target_scores_segm = [self.target_model_segm.apply_target_model(f, train_feat_clf) for f in filter_iter_segm]
        target_scores_last_iter_segm = target_scores_segm[-1]
        coarse_mask = torch.cat((train_bbox_enc, target_scores_last_iter_segm), dim=2)
        pred_all_segm, _ = self.decoder(coarse_mask, train_feat, train_imgs.shape[-2:])
        pred_all_segm = pred_all_segm.view(num_train_frames, num_sequences, *pred_all_segm.shape[-2:])

        return pred_all, pred_all_segm

    def segment_target_add_bbox_encoder(self, bbox_mask, target_filter, test_feat_clf, test_feat, segm):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        
        if not segm:
            target_scores = self.target_model.apply_target_model(target_filter, test_feat_clf)
        else:
            target_scores = self.target_model_segm.apply_target_model(target_filter, test_feat_clf)

        target_scores = torch.cat((bbox_mask, target_scores), dim=2)
        mask_pred, decoder_feat = self.decoder(target_scores, test_feat,
                                               (test_feat_clf.shape[-2]*16, test_feat_clf.shape[-1]*16))
        # Output is 1, 1, h, w
        return mask_pred

    def get_backbone_target_model_features(self, backbone_feat):
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
                              backbone_type='imagenet',):
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

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)
    initializer_segm = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                             feature_dim=out_feature_dim, filter_groups=filter_groups)
    
    # Few-shot label generator and weight predictor
    label_encoder = seg_label_encoder.ResidualDS16FeatSWBox(layer_dims=label_encoder_dims + (num_filters, ),
                                                            feat_dim=out_feature_dim, use_final_relu=True,
                                                            use_gauss=False)
    bbox_encoder = seg_label_encoder.ResidualDS16FeatSWBox(layer_dims=label_encoder_dims + (num_filters, ),
                                                           feat_dim=out_feature_dim, use_final_relu=True,
                                                           use_gauss=False)
    segm_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims[:-1] + (num_filters, ), 
                                                    use_bn=use_bn_in_label_enc)

    # Computes few-shot learning loss
    residual_module = loss_residual_modules.LWTLResidual(init_filter_reg=optim_init_reg)
    residual_module_segm = loss_residual_modules.LWTLResidual(init_filter_reg=optim_init_reg)

    # Iteratively updates the target model parameters by minimizing the few-shot learning loss
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, 
                                                  detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)
    optimizer_segm = steepestdescent.GNSteepestDescent(residual_module=residual_module_segm, num_iter=optim_iter, 
                                                       detach_length=detach_length,
                                                       residual_batch_dim=1, compute_losses=True)

    # Target model and Few-shot learner
    target_model = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                           filter_optimizer=optimizer, feature_extractor=target_model_feature_extractor,
                                           filter_dilation_factors=dilation_factors)
    target_model_segm = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer_segm,
                                                filter_optimizer=optimizer_segm, feature_extractor=None,
                                                filter_dilation_factors=dilation_factors)

    # Decoder
    decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

    decoder = lwtl_decoder.LWTLDecoder(num_filters*2, decoder_mdim, decoder_input_layers_channels, use_bn=True)

    net = STANet(feature_extractor=backbone_net, target_model=target_model, target_model_segm=target_model_segm,
                 decoder=decoder,
                 label_encoder=label_encoder, bbox_encoder=bbox_encoder, segm_encoder=segm_encoder,
                 target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers)
    return net
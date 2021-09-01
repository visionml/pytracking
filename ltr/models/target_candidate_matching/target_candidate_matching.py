import torch
import torch.utils.checkpoint

from torch import nn

from collections import OrderedDict

from ltr import model_constructor
import ltr.models.backbone as backbones
from ltr.models.target_candidate_matching.superglue import SuperGlue


class DescriptorExtractor(nn.Module):
    def __init__(self, backbone_feat_dim, descriptor_dim, kernel_size=4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=backbone_feat_dim, out_channels=descriptor_dim, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=True)

    def forward(self, x, coords):
        feats =  self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0,2,1)

    def get_descriptors(self, x, coords):
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)

        feats = self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0, 2, 1)


class TargetCandidateMatchingNetwork(nn.Module):
    def __init__(self, feature_extractor, classification_layer, descriptor_extractor, matcher):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_layer = classification_layer
        self.output_layers = sorted(list(set(self.classification_layer)))

        self.descriptor_extractor = descriptor_extractor
        self.matcher = matcher


    def forward(self, img_cropped0, img_cropped1, candidate_tsm_coords0, candidate_tsm_coords1, candidate_img_coords0,
                candidate_img_coords1, candidate_scores0, candidate_scores1, img_shape0, img_shape1, **kwargs):


        # Extract backbone features
        frame_feat0 = self.extract_backbone_features(img_cropped0.reshape(-1, *img_cropped0.shape[-3:]))
        frame_feat1 = self.extract_backbone_features(img_cropped1.reshape(-1, *img_cropped1.shape[-3:]))

        # Classification features
        frame_feat_clf0 = self.get_backbone_clf_feat(frame_feat0)
        frame_feat_clf1 = self.get_backbone_clf_feat(frame_feat1)

        descriptors0 = self.descriptor_extractor(frame_feat_clf0, candidate_tsm_coords0[0])
        descriptors1 = self.descriptor_extractor(frame_feat_clf1, candidate_tsm_coords1[0])

        data = {
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
            'img_coords0': candidate_img_coords0[0],
            'img_coords1': candidate_img_coords1[0],
            'scores0': candidate_scores0[0],
            'scores1': candidate_scores1[0],
            'image_size0': img_shape0[0],
            'image_size1': img_shape1[0],
        }

        pred = self.matcher(data)

        return pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat


@model_constructor
def target_candidate_matching_net_resnet50(backbone_pretrained=True, classification_layer=None,
                                           frozen_backbone_layers=(), skip_gnn=False, GNN_layers=None,
                                           num_sinkhorn_iterations=10, output_normalization='sinkhorn'):
    if classification_layer is None:
        classification_layer = ['layer3']
    if GNN_layers is None:
        GNN_layers = ['self', 'cross'] * 2

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    descriptor_extractor = DescriptorExtractor(backbone_feat_dim=1024, descriptor_dim=256, kernel_size=4)

    conf = {
        'skip_gnn': skip_gnn,
        'GNN_layers': GNN_layers,
        'num_sinkhorn_iterations': num_sinkhorn_iterations,
        'output_normalization': output_normalization
    }

    matcher = SuperGlue(conf=conf)

    net = TargetCandidateMatchingNetwork(feature_extractor=backbone_net, classification_layer=classification_layer,
                                         descriptor_extractor=descriptor_extractor, matcher=matcher)

    return net

from . import BaseActor
import torch
import torch.nn as nn
import numpy as np
from pytracking.analysis.vos_utils import davis_jaccard_measure


class LWLActor(BaseActor):
    """Actor for training the LWL network."""
    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        """
        args:
            net - The network model to train
            objective - Loss functions
            loss_weight - Weights for each training loss
            num_refinement_iter - Number of update iterations N^{train}_{update} used to update the target model in
                                  each frame
            disable_backbone_bn - If True, all batch norm layers in the backbone feature extractor are disabled, i.e.
                                  set to eval mode.
            disable_all_bn - If True, all the batch norm layers in network are disabled, i.e. set to eval mode.
        """
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_masks',
                    'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss_segm.item(),
                 'Stats/acc': acc / cnt}

        return loss, stats


class LWLBoxActor(BaseActor):
    """Actor for training bounding box encoder """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'train_anno', and 'train_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        train_imgs = data['train_images']
        bb_train = data['train_anno']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.net.extract_target_model_features(train_feat)  # seq*frames, channels, height, width
        bb_train = bb_train.view(-1, *bb_train.shape[-1:])
        train_box_enc = self.net.box_label_encoder(bb_train, train_feat_clf, train_imgs.shape)
        train_box_enc = train_box_enc.view(num_train_frames, num_sequences, *train_box_enc.shape[-3:])

        mask_pred_box_train, decoder_feat_train = self.net.decoder(train_box_enc, train_feat, train_imgs.shape[-2:])

        loss_segm_box = self.loss_weight['segm_box'] * self.objective['segm'](mask_pred_box_train, data['train_masks'].view(mask_pred_box_train.shape))
        loss_segm_box = loss_segm_box / num_train_frames
        stats = {}

        loss = loss_segm_box

        acc_box = 0
        cnt_box = 0
        acc_lbox = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(mask_pred_box_train.view(-1, *mask_pred_box_train.shape[-2:]), data['train_masks'].view(-1, *mask_pred_box_train.shape[-2:]))]
        acc_box += sum(acc_lbox)
        cnt_box += len(acc_lbox)

        stats['Loss/total'] = loss.item()
        stats['Stats/acc_box_train'] = acc_box/cnt_box

        return loss, stats


class RTSActor(BaseActor):
    """Based on the LWL and DiMP actor, supporting both segmentation and classification terms """
    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        """
        args:
            net - The network model to train
            objective - Loss functions
            loss_weight - Weights for each training loss
            num_refinement_iter - Number of update iterations N^{train}_{update} used to update the target model in
                                  each frame
            disable_backbone_bn - If True, all batch norm layers in the backbone feature extractor are disabled, i.e.
                                  set to eval mode.
            disable_all_bn - If True, all the batch norm layers in network are disabled, i.e. set to eval mode.
        """
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_masks',
                    'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        segm_pred, target_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_masks=data['train_masks'],
                                            test_masks=data['test_masks'],
                                            train_bb=data['train_anno'],
                                            train_label=data['train_label'],
                                            test_label=data['test_label'],
                                            num_refinement_iter=self.num_refinement_iter)

        # Segmentation Loss
        ################################################################
        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)


        if torch.isinf(loss_segm) or torch.isnan(loss_segm):
            raise Exception('ERROR: Segm Loss was nan or inf!!!')


        # Classification Loss
        #################################################################

        threshold = 0.25

        last_target_score = target_scores[-1]
        label = data['test_label']

        last_target_score_reshaped = last_target_score.view(-1, last_target_score.shape[-2] * last_target_score.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])
        prediction_max_vals, pred_argmax_ids = last_target_score_reshaped.max(dim=1)
        label_max_vals, label_argmax_ids = label_reshaped.max(dim=1)

        label_val_at_peak = label_reshaped[torch.arange(len(pred_argmax_ids)), pred_argmax_ids]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))

        prediction_correct = ((label_val_at_peak >= threshold) & (label_max_vals > 0.25)) | ((label_val_at_peak < threshold) & (label_max_vals < 0.25))
        prediction_accuracy = prediction_correct.float().mean()

        # Peak to Peak distance
        n_samples = label.shape[0] * label.shape[1]
        n_pixels = label.shape[2] * label.shape[3]

        peak_dist = np.zeros(n_samples)

        assert(last_target_score.shape == label.shape)

        for sample in range(0, n_samples):
            pred_idx = pred_argmax_ids[sample].cpu()
            label_idx = label_argmax_ids[sample].cpu()
            peak_pred = np.array([pred_idx // label.shape[-1], pred_idx % label.shape[-1]])
            peak_label = np.array([label_idx // label.shape[-1], label_idx % label.shape[-1]])
            peak_dist[sample] = np.linalg.norm(peak_label - peak_pred)

        peak_dist = peak_dist.mean()

        # Compute loss
        loss_test_init_clf = 0
        loss_test_iter_clf = 0

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](
            s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Loss for the initial filter iteration
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss_classifier = loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss_classifier) or torch.isnan(loss_classifier):
            raise Exception('ERROR: Classifier Loss was nan or inf!!!')

        # TOTAL LOSS
        loss = loss_segm + loss_classifier

        # Log stats Segmentation
        stats = {
            'Loss/total': loss.item(),
            'Loss/segm': loss_segm.item(),
            'Stats/acc': acc / cnt,
            'Stats/clf_acc': prediction_accuracy,
            'Stats/clf_peak_dist': peak_dist,
        }

        # Log stats Classification
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


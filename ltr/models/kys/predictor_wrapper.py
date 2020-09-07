import torch
import torch.nn as nn
from ltr.models.kys.utils import shift_features


class PredictorWrapper(nn.Module):
    def __init__(self, cost_volume, predictor):
        super().__init__()

        self.cost_volume = cost_volume
        self.predictor = predictor
        self.fix_coordinate_shift = True

    def forward(self, data):
        input1 = data['input1']
        input2 = data['input2']
        label_prev = data.get('label_prev', None)

        dimp_score_cur = data['dimp_score_cur']
        state_prev = data['state_prev']

        score_shape = dimp_score_cur.shape

        if isinstance(input1, (tuple, list)):
            feat1 = [self.extract_motion_feat(in1) for in1 in input1]
            feat1 = [f1.view(-1, *f1.shape[-3:]) for f1 in feat1]
        else:
            feat1 = self.extract_motion_feat(input1)
            feat1 = feat1.view(-1, *feat1.shape[-3:])

        feat2 = self.extract_motion_feat(input2)
        feat2 = feat2.view(-1, *feat2.shape[-3:])

        dimp_score_cur = dimp_score_cur.view(-1, 1, *dimp_score_cur.shape[-2:])

        if isinstance(input1, (tuple, list)):
            cost_volume = [self.compute_cost_volume(f1, feat2, True) for f1 in feat1]
        else:
            cost_volume = self.compute_cost_volume(feat1, feat2, True)

        feat_map_size = torch.tensor([dimp_score_cur.shape[-1], dimp_score_cur.shape[-2]]).view(1, 2).float().to(dimp_score_cur.device)
        if self.fix_coordinate_shift:
            shift_value = - torch.ones(dimp_score_cur.shape[0], 2).to(dimp_score_cur.device) * 0.5 / feat_map_size

            if label_prev is not None:
                label_prev_shape = label_prev.shape
                label_prev = shift_features(label_prev.clone().view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(label_prev_shape)

            dimp_score_cur = shift_features(dimp_score_cur.clone(), shift_value)

        pred_response, state_new, auxiliary_outputs = self.predictor(cost_volume, state_prev, dimp_score_cur, label_prev)
        pred_response = pred_response.view(score_shape)

        # Shift back
        if self.fix_coordinate_shift:
            shift_value = torch.ones(dimp_score_cur.shape[0], 2).to(dimp_score_cur.device) * 0.5 / feat_map_size

            if 'is_target' in auxiliary_outputs:
                auxiliary_outputs['is_target'] = shift_features(
                    auxiliary_outputs['is_target'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)

            if 'is_target_after_prop' in auxiliary_outputs:
                auxiliary_outputs['is_target_after_prop'] = shift_features(
                    auxiliary_outputs['is_target_after_prop'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)

            if 'is_target_new' in auxiliary_outputs:
                auxiliary_outputs['is_target_new'] = shift_features(
                    auxiliary_outputs['is_target_new'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)

            pred_response = shift_features(pred_response.view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)

        output = {'response': pred_response, 'state_cur': state_new, 'auxiliary_outputs': auxiliary_outputs}
        return output

    def compute_cost_volume(self, feat_prev, feat_cur, use_current_frame_as_ref):
        if use_current_frame_as_ref:
            cost_volume = self.cost_volume(feat_cur, feat_prev)
        else:
            cost_volume = self.cost_volume(feat_prev, feat_cur)
        return cost_volume

    def extract_motion_feat(self, backbone_feat):
        backbone_feat = backbone_feat.view(-1, backbone_feat.shape[-3], backbone_feat.shape[-2],
                                           backbone_feat.shape[-1])

        return backbone_feat

    def predict_response(self, data, dimp_thresh=None, output_window=None):
        feat1 = data['feat1']
        feat2 = data['feat2']
        label_prev = data.get('label_prev', None)

        dimp_score_cur = data['dimp_score_cur']
        state_prev = data['state_prev']

        score_shape = dimp_score_cur.shape

        if isinstance(feat1, (tuple, list)):
            feat1 = [f1.view(-1, *f1.shape[-3:]) for f1 in feat1]
        else:
            feat1 = feat1.view(-1, *feat1.shape[-3:])

        feat2 = feat2.view(-1, *feat2.shape[-3:])
        dimp_score_cur = dimp_score_cur.view(-1, 1, *dimp_score_cur.shape[-2:])

        if isinstance(feat1, (tuple, list)):
            cost_volume = [self.compute_cost_volume(f1, feat2, True) for f1 in feat1]
        else:
            cost_volume = self.compute_cost_volume(feat1, feat2, True)

        feat_map_size = torch.tensor([dimp_score_cur.shape[-1], dimp_score_cur.shape[-2]]).view(1, 2).float().to(
            dimp_score_cur.device)
        if self.fix_coordinate_shift:
            shift_value = - torch.ones(dimp_score_cur.shape[0], 2).to(dimp_score_cur.device) * 0.5 / feat_map_size

            if label_prev is not None:
                label_prev_shape = label_prev.shape
                label_prev = shift_features(label_prev.clone().view(-1, 1, *dimp_score_cur.shape[-2:]),
                                            shift_value).view(label_prev_shape)

            dimp_score_cur = shift_features(dimp_score_cur.clone(), shift_value)

        pred_response, state_new, auxiliary_outputs = self.predictor(cost_volume, state_prev, dimp_score_cur,
                                                                     label_prev, dimp_thresh, output_window)

        pred_response = pred_response.view(score_shape)

        # Shift back
        if self.fix_coordinate_shift:
            shift_value = torch.ones(dimp_score_cur.shape[0], 2).to(dimp_score_cur.device) * 0.5 / feat_map_size

            if 'is_target' in auxiliary_outputs:
                auxiliary_outputs['is_target'] = shift_features(
                    auxiliary_outputs['is_target'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(
                    score_shape)

            if 'is_target_after_prop' in auxiliary_outputs:
                auxiliary_outputs['is_target_after_prop'] = shift_features(
                    auxiliary_outputs['is_target_after_prop'].view(-1, 1, *dimp_score_cur.shape[-2:]),
                    shift_value).view(score_shape)

            if 'is_target_new' in auxiliary_outputs:
                auxiliary_outputs['is_target_new'] = shift_features(
                    auxiliary_outputs['is_target_new'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(
                    score_shape)

            pred_response = shift_features(pred_response.view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(
                score_shape)

        output = {'response': pred_response, 'state_cur': state_new, 'auxiliary_outputs': auxiliary_outputs,
                  'cost_volume': cost_volume}

        return output
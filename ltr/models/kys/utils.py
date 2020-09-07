import torch
import torch.nn as nn
import torch.nn.functional as F


def shift_features(feat, relative_translation_vector):
    T_mat = torch.eye(2).repeat(feat.shape[0], 1, 1).to(feat.device)
    T_mat = torch.cat((T_mat, relative_translation_vector.view(-1, 2, 1)), dim=2)

    grid = F.affine_grid(T_mat, feat.shape)

    feat_out = F.grid_sample(feat, grid)
    return feat_out


class CenterShiftFeatures(nn.Module):
    def __init__(self, feature_stride):
        super().__init__()
        self.feature_stride = feature_stride

    def forward(self, feat, anno):
        anno = anno.view(-1, 4)
        c_x = (anno[:, 0] + anno[:, 2] * 0.5) / self.feature_stride
        c_y = (anno[:, 1] + anno[:, 3] * 0.5) / self.feature_stride

        t_x = 2 * (c_x - feat.shape[-1] * 0.5) / feat.shape[-1]
        t_y = 2 * (c_y - feat.shape[-2] * 0.5) / feat.shape[-2]

        t = torch.cat((t_x.view(-1, 1), t_y.view(-1, 1)), dim=1)

        feat_out = shift_features(feat, t)
        return feat_out


class DiMPScoreJittering():
    def __init__(self, p_zero=0.0, distractor_ratio=1.0, p_distractor=0, max_distractor_enhance_factor=1,
                 min_distractor_enhance_factor=0.75):
        """ Jitters predicted score map by randomly enhancing distractor peaks and masking out target peaks"""
        self.p_zero = p_zero
        self.distractor_ratio = distractor_ratio
        self.p_distractor = p_distractor
        self.max_distractor_enhance_factor = max_distractor_enhance_factor
        self.min_distractor_enhance_factor = min_distractor_enhance_factor

    def rand(self, sz, min_val, max_val):
        return torch.rand(sz, device=min_val.device) * (max_val - min_val) + min_val

    def __call__(self, score, label):
        score_shape = score.shape

        score = score.view(-1, score_shape[-2]*score_shape[-1])
        num_score_maps = score.shape[0]

        label = label.view(score.shape)

        dist_roll_value = torch.rand(num_score_maps).to(score.device)

        score_c = score.clone().detach()
        score_neg = score_c * (label < 1e-4).float()
        score_pos = score_c * (label > 0.2).float()

        target_max_val, _ = torch.max(score_pos, dim=1)
        dist_max_val, dist_id = torch.max(score_neg, dim=1)

        jitter_score = (dist_roll_value < self.p_distractor) & ((dist_max_val / target_max_val) > self.distractor_ratio)

        for i in range(num_score_maps):
            score_c[i, dist_id[i]] = self.rand(1, target_max_val[i]*self.min_distractor_enhance_factor,
                                               target_max_val[i]*self.max_distractor_enhance_factor)

        zero_roll_value = torch.rand(num_score_maps).to(score.device)
        zero_score = (zero_roll_value < self.p_zero) & ~jitter_score

        score_c[zero_score, :] = 0

        score_jittered = score*(1.0 - (jitter_score | zero_score).float()).view(num_score_maps, 1).float() + \
                         score_c*(jitter_score | zero_score).float().view(num_score_maps, 1).float()

        return score_jittered.view(score_shape)

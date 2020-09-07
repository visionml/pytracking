import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.models.layers.blocks import conv_block
from .conv_gru import ConvGRUCell


class ResponsePredictor(nn.Module):
    def __init__(self, state_dim=8, representation_predictor_dims=(64, 32), gru_ksz=3,
                 prev_max_pool_ksz=1, conf_measure='max', dimp_thresh=None):
        super().__init__()
        self.prev_max_pool_ksz = prev_max_pool_ksz
        self.conf_measure = conf_measure
        self.dimp_thresh = dimp_thresh

        cvproc_ksz = [3, 3]
        use_bn = True

        padding_val = [int((s - 1) / 2) for s in cvproc_ksz]

        self.cost_volume_proc1 = nn.Sequential(
            conv_block(1, 8, kernel_size=cvproc_ksz[0], stride=1, padding=padding_val[0], batch_norm=use_bn, relu=True),
            conv_block(8, 1, kernel_size=cvproc_ksz[1], stride=1, padding=padding_val[1], batch_norm=use_bn, relu=False))

        self.cost_volume_proc2 = nn.Sequential(
            conv_block(1, 8, kernel_size=cvproc_ksz[0], stride=1, padding=padding_val[0], batch_norm=use_bn, relu=True),
            conv_block(8, 1, kernel_size=cvproc_ksz[1], stride=1, padding=padding_val[1], batch_norm=use_bn, relu=False))

        in_dim = state_dim + 1 + (conf_measure != 'none')
        representation_predictor_list = []
        for out_dim in representation_predictor_dims:
            representation_predictor_list.append(conv_block(in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                                                            batch_norm=False, relu=True))
            in_dim = out_dim

        self.representation_predictor = nn.Sequential(*representation_predictor_list)
        self.representation_dim = in_dim

        self.response_predictor = nn.Sequential(
            conv_block(in_dim, 1, kernel_size=3, stride=1, padding=1, batch_norm=False, relu=False),
            nn.Sigmoid())

        self.state_predictor = ConvGRUCell(4, state_dim, gru_ksz)

        self.init_hidden_state_predictor = nn.Sequential(
                conv_block(1, state_dim, kernel_size=3, stride=1, padding=1, batch_norm=False, relu=False, bias=False),
                nn.Tanh())

        self.is_target_predictor = nn.Sequential(
            conv_block(state_dim, 4, kernel_size=gru_ksz, stride=1, padding=int(gru_ksz // 2), batch_norm=False,
                       relu=True),
            conv_block(4, 1, kernel_size=gru_ksz, stride=1, padding=int(gru_ksz // 2), batch_norm=False, relu=False))

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, cost_volume, state_prev, dimp_score_cur, init_label=None, dimp_thresh=None,
                output_window=None):
        # Cost vol shape: n x h*w x h x w
        # state_prev shape: n x d x h x w
        # dimp_cur_shape: n x 1 x h x w
        # init_label shape: n x 1 x h x w
        if dimp_thresh is None:
            dimp_thresh = self.dimp_thresh
        auxiliary_outputs = {}

        num_sequences = cost_volume.shape[0]
        feat_sz = cost_volume.shape[-2:]

        cost_volume = cost_volume.view(-1, 1, feat_sz[0], feat_sz[1])
        cost_volume_p1 = self.cost_volume_proc1(cost_volume).view(-1, feat_sz[0] * feat_sz[1])
        cost_volume_p1 = F.softmax(cost_volume_p1, dim=1)

        cost_volume_p2 = self.cost_volume_proc2(cost_volume_p1.view(-1, 1, feat_sz[0], feat_sz[1]))
        cost_volume_p2 = cost_volume_p2.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        cost_volume_p2 = F.softmax(cost_volume_p2, dim=1)
        cost_volume_p2 = cost_volume_p2.view(num_sequences, -1, 1, feat_sz[0], feat_sz[1])

        auxiliary_outputs['cost_volume_processed'] = cost_volume_p2

        if state_prev is None:
            init_hidden_state = self.init_hidden_state_predictor(init_label.view(num_sequences, 1,
                                                                                 feat_sz[0], feat_sz[1]))
            state_prev_ndhw = init_hidden_state
        else:
            state_prev_ndhw = state_prev

        is_target = self.is_target_predictor(state_prev_ndhw)
        auxiliary_outputs['is_target'] = is_target

        state_prev_ndhw = state_prev_ndhw.view(num_sequences, -1, feat_sz[0], feat_sz[1])

        state_prev_nhwd = state_prev_ndhw.permute(0, 2, 3, 1).contiguous(). \
            view(num_sequences, feat_sz[0] * feat_sz[1], -1, 1, 1).expand(-1, -1, -1, feat_sz[0], feat_sz[1])

        #  Compute propagation weights
        propagation_weight_norm = cost_volume_p2.view(num_sequences, feat_sz[0] * feat_sz[1], 1, feat_sz[0], feat_sz[1])

        # Pool
        if self.prev_max_pool_ksz > 1:
            raise NotImplementedError

        # Max pool along prev frame ref
        if self.conf_measure == 'max':
            propagation_conf = propagation_weight_norm.view(num_sequences, -1, feat_sz[0], feat_sz[1]).max(dim=1)[0]
        elif self.conf_measure == 'entropy':
            propagation_conf = propagation_weight_norm.view(num_sequences, -1, feat_sz[0], feat_sz[1])
            propagation_conf = -(propagation_conf * (propagation_conf + 1e-4).log()).sum(dim=1)

        auxiliary_outputs['propagation_weights'] = propagation_weight_norm

        propagated_h = (propagation_weight_norm * state_prev_nhwd).sum(dim=1)
        propagated_h = propagated_h.view(num_sequences, -1, feat_sz[0], feat_sz[1])

        auxiliary_outputs['propagated_h'] = propagated_h.clone()
        is_target_after_prop = self.is_target_predictor(propagated_h)
        auxiliary_outputs['is_target_after_prop'] = is_target_after_prop

        if self.conf_measure != 'none':
            propagation_conf = propagation_conf.view(num_sequences, 1, feat_sz[0], feat_sz[1])
            auxiliary_outputs['propagation_conf'] = propagation_conf

            predictor_input = torch.cat(
                    (propagated_h, dimp_score_cur.view(num_sequences, 1, *dimp_score_cur.shape[-2:]),
                     propagation_conf), dim=1)
        else:
            predictor_input = torch.cat(
                (propagated_h, dimp_score_cur.view(num_sequences, 1, *dimp_score_cur.shape[-2:])), dim=1)
        resp_representation = self.representation_predictor(predictor_input)
        fused_prediction = self.response_predictor(resp_representation)

        auxiliary_outputs['fused_score_orig'] = fused_prediction.clone()
        if dimp_thresh is not None:
            fused_prediction = fused_prediction * (dimp_score_cur > dimp_thresh).float()

        if output_window is not None:
            fused_prediction = fused_prediction * output_window

        scores_cat = torch.cat((dimp_score_cur, fused_prediction), dim=1)

        scores_cat_pool = F.adaptive_max_pool2d(scores_cat, 1).view(scores_cat.shape[0], scores_cat.shape[1], 1, 1). \
            expand(-1, -1, feat_sz[0], feat_sz[1])

        state_gru_input = torch.cat((scores_cat, scores_cat_pool), dim=1)

        state_new = self.state_predictor(state_gru_input, propagated_h)

        is_target_new = self.is_target_predictor(state_new)
        auxiliary_outputs['is_target_new'] = is_target_new

        return fused_prediction, state_new, auxiliary_outputs


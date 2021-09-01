import torch
import torch.nn as nn


def recall(m, gt_m):
    mask = (gt_m > -1).float()
    return ((m == gt_m) * mask).sum(1) / mask.sum(1)


def precision(m, gt_m):
    mask = ((m > -1) & (gt_m >= -1)).float()
    prec = ((m == gt_m) * mask).sum(1) / torch.max(mask.sum(1), torch.ones_like(mask.sum(1)))
    no_match_mask = (gt_m > -1).sum(1) == 0
    prec[no_match_mask] = float('NaN')
    return prec


class TargetCandidateMatchingLoss(nn.Module):
    def __init__(self, nll_balancing=0.5, nll_weight=1.):
        super().__init__()
        self.nll_balancing = nll_balancing
        self.nll_weight = nll_weight


    def metrics(self, matches1, gt_matches1, **kwargs):
        rec = recall(matches1, gt_matches1[0])
        prec = precision(matches1, gt_matches1[0])
        return {'match_recall': rec, 'match_precision': prec}

    def forward(self, gt_assignment, gt_matches0, gt_matches1, log_assignment, bin_score, **kwargs):
        gt_assignment = gt_assignment[0]
        gt_matches0 = gt_matches0[0]
        gt_matches1 = gt_matches1[0]

        losses = {'total': 0}

        positive = gt_assignment.float()
        neg0 = (gt_matches0 == -1).float()
        neg1 = (gt_matches1 == -1).float()

        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))

        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg

        nll = (self.nll_balancing * nll_pos + (1 - self.nll_balancing) * nll_neg)

        losses['assignment_nll'] = nll

        if self.nll_weight > 0:
            losses['total'] = nll * self.nll_weight

        # Some statistics
        losses['nll_pos'] = nll_pos
        losses['nll_neg'] = nll_neg
        losses['num_matchable'] = num_pos
        losses['num_unmatchable'] = num_neg
        losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
        losses['bin_score'] = bin_score[None]

        return losses

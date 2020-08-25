import math
import torch
import torch.nn as nn
from pytracking import TensorList
from ltr.models.layers import activation


class GNSteepestDescent(nn.Module):
    """General module for steepest descent based meta learning."""
    def __init__(self, residual_module, num_iter=1, compute_losses=False, detach_length=float('Inf'),
                 parameter_batch_dim=0, residual_batch_dim=0, steplength_reg=0.0):
        super().__init__()

        self.residual_module = residual_module
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self._parameter_batch_dim = parameter_batch_dim
        self._residual_batch_dim = residual_batch_dim


    def _sqr_norm(self, x: TensorList, batch_dim=0):
        sum_keep_batch_dim = lambda e: e.sum(dim=[d for d in range(e.dim()) if d != batch_dim])
        return sum((x * x).apply(sum_keep_batch_dim))


    def _compute_loss(self, res):
        return sum((res * res).sum()) / sum(res.numel())


    def forward(self, meta_parameter: TensorList, num_iter=None, *args, **kwargs):
        input_is_list = True
        if not isinstance(meta_parameter, TensorList):
            meta_parameter = TensorList([meta_parameter])
            input_is_list = False


        # Make sure grad is enabled
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        num_iter = self.num_iter if num_iter is None else num_iter

        meta_parameter_iterates = []
        def _add_iterate(meta_par):
            if input_is_list:
                meta_parameter_iterates.append(meta_par)
            else:
                meta_parameter_iterates.append(meta_par[0])

        _add_iterate(meta_parameter)

        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()

            meta_parameter.requires_grad_(True)

            # Compute residual vector
            r = self.residual_module(meta_parameter, **kwargs)

            if self.compute_losses:
                losses.append(self._compute_loss(r))

            # Compute gradient of loss
            u = r.clone()
            g = TensorList(torch.autograd.grad(r, meta_parameter, u, create_graph=True))

            # Multiply gradient with Jacobian
            h = TensorList(torch.autograd.grad(g, u, g, create_graph=True))

            # Compute squared norms
            ip_gg = self._sqr_norm(g, batch_dim=self._parameter_batch_dim)
            ip_hh = self._sqr_norm(h, batch_dim=self._residual_batch_dim)

            # Compute step length
            alpha = ip_gg / (ip_hh + self.steplength_reg * ip_gg).clamp(1e-8)

            # Compute optimization step
            step = g.apply(lambda e: alpha.reshape([-1 if d==self._parameter_batch_dim else 1 for d in range(e.dim())]) * e)

            # Add step to parameter
            meta_parameter = meta_parameter - step

            _add_iterate(meta_parameter)

        if self.compute_losses:
            losses.append(self._compute_loss(self.residual_module(meta_parameter, **kwargs)))

        # Reset the grad enabled flag
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()

        if not input_is_list:
            meta_parameter = meta_parameter[0]

        return meta_parameter, meta_parameter_iterates, losses


class KLRegSteepestDescent(nn.Module):
    """General meta learning module for Steepest Descent based meta learning with Newton when minimizing KL-divergence."""
    def __init__(self, score_predictor, num_iter=1, compute_losses=True, detach_length=float('Inf'),
                 parameter_batch_dim=0, steplength_reg=0.0, hessian_reg=0, init_step_length=1.0,
                 softmax_reg=None):
        super().__init__()

        self.score_predictor = score_predictor
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self.hessian_reg = hessian_reg
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.softmax_reg = softmax_reg
        self._parameter_batch_dim = parameter_batch_dim


    def forward(self, meta_parameter: TensorList, num_iter=None, **kwargs):
        if not isinstance(meta_parameter, TensorList):
            meta_parameter = TensorList([meta_parameter])

        _residual_batch_dim = 1

        # Make sure grad is enabled
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        num_iter = self.num_iter if num_iter is None else num_iter

        step_length_factor = torch.exp(self.log_step_length)

        label_density, sample_weight, reg_weight = self.score_predictor.init_data(meta_parameter, **kwargs)

        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

        def _compute_loss(scores, weights):
            num_sequences = scores.shape[_residual_batch_dim]
            return torch.sum(sample_weight.reshape(sample_weight.shape[0], -1) *
                             (torch.log(scores.exp().sum(dim=(-2, -1)) + exp_reg) - (label_density * scores).sum(dim=(-2, -1)))) / num_sequences + \
                   reg_weight * sum((weights * weights).sum()) / num_sequences

        meta_parameter_iterates = [meta_parameter]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()

            meta_parameter.requires_grad_(True)

            # Compute residual vector
            scores = self.score_predictor(meta_parameter, **kwargs)

            if self.compute_losses:
                losses.append(_compute_loss(scores, meta_parameter))

            scores_softmax = activation.softmax_reg(scores.reshape(*scores.shape[:2], -1), dim=2,
                                                    reg=self.softmax_reg).reshape(scores.shape)
            dLds = sample_weight * (scores_softmax - label_density)

            # Compute gradient of loss
            weights_grad = TensorList(torch.autograd.grad(scores, meta_parameter, dLds, create_graph=True)) + \
                          meta_parameter * reg_weight

            # Multiply gradient with Jacobian
            scores_grad = torch.autograd.grad(weights_grad, dLds, weights_grad, create_graph=True)[0]

            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(sm_scores_grad, dim=(-2, -1), keepdim=True) + \
                              self.hessian_reg * scores_grad
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(*scores.shape[:2], -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (sample_weight.reshape(sample_weight.shape[0], -1) * grad_hes_grad).sum(dim=0)

            # Compute optimal step length
            gg = (weights_grad * weights_grad).reshape(scores.shape[1], -1).sum(dim=1)
            alpha_num = sum(gg)
            alpha_den = (grad_hes_grad + sum(gg * reg_weight) + self.steplength_reg * alpha_num).clamp(1e-8)
            alpha = step_length_factor * (alpha_num / alpha_den)

            # Compute optimization step
            step = weights_grad.apply(
                lambda e: alpha.reshape([-1 if d == self._parameter_batch_dim else 1 for d in range(e.dim())]) * e)

            # Add step to parameter
            meta_parameter = meta_parameter - step

            meta_parameter_iterates.append(meta_parameter)

        if self.compute_losses:
            losses.append(_compute_loss(self.score_predictor(meta_parameter, **kwargs), meta_parameter))

        # Reset the grad enabled flag
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()

        return meta_parameter, meta_parameter_iterates, losses

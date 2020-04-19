import math
import torch
import torch.nn as nn
from pytracking import TensorList
from ltr.models.layers import activation


class GNSteepestDescent(nn.Module):
    """General module for steepest descent based meta learning."""
    def __init__(self, residual_module, num_iter=1, compute_losses=False, detach_length=float('Inf'),
                 parameter_batch_dim=0, residual_batch_dim=0, steplength_reg=0.0,
                 filter_dilation_factors=None):
        super().__init__()

        self.residual_module = residual_module
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self._parameter_batch_dim = parameter_batch_dim
        self._residual_batch_dim = residual_batch_dim
        self.filter_dilation_factors = filter_dilation_factors

    def _sqr_norm(self, x: TensorList, batch_dim=0):
        sum_keep_batch_dim = lambda e: e.sum(dim=[d for d in range(e.dim()) if d != batch_dim])
        return sum((x * x).apply(sum_keep_batch_dim))


    def _compute_loss(self, res):
        return sum((res * res).sum()) / sum(res.numel())


    def forward(self, meta_parameter: TensorList, num_iter=None, *args, **kwargs):
        # Make sure grad is enabled
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        num_iter = self.num_iter if num_iter is None else num_iter

        meta_parameter_iterates = [meta_parameter]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()

            meta_parameter.requires_grad_(True)

            # Compute residual vector
            r = self.residual_module(meta_parameter, filter_dilation_factors=self.filter_dilation_factors, **kwargs)

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

            meta_parameter_iterates.append(meta_parameter)


        if self.compute_losses:
            losses.append(self._compute_loss(self.residual_module(meta_parameter,
                                                                  filter_dilation_factors=self.filter_dilation_factors,
                                                                  **kwargs)))

        # Reset the grad enabled flag
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()

        return meta_parameter, meta_parameter_iterates, losses

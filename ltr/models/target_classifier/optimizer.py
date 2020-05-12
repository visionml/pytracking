import torch.nn as nn
import torch
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
import math



class DiMPSteepestDescentGN(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """
    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 score_act='relu', act_param=None, min_filter_reg=1e-3, mask_act='sigmoid',
                 detach_length=float('Inf'), alpha_eps=0):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        # Module that predicts the target label function (y in the paper)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        # Module that predicts the target mask (m in the paper)
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias

        # Module that predicts the residual weights (v in the paper)
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # The score actvation and its derivative
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')


    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1) * spatial_weight

        backprop_through_learning = (self.detach_length > 0)

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if not backprop_through_learning or (i > 0 and i % self.detach_length == 0):
                weights = weights.detach()

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)

            if compute_losses:
                losses.append(((residuals**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

            # Compute gradient
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + \
                          reg_weight * weights

            # Map the gradient with the Jacobian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).sum(dim=(1,2,3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (step_length_factor * alpha.reshape(-1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = self.score_activation(scores, target_mask)
            losses.append((((sample_weight * (scores - label_map))**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

        return weights, weight_iterates, losses



class DiMPL2SteepestDescentGN(nn.Module):
    """A simpler optimizer module that uses L2 loss.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        gauss_sigma:  The standard deviation of the label function.
        hinge_threshold:  Threshold for the hinge-based loss (see DiMP paper).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """
    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, gauss_sigma=1.0, hinge_threshold=-999,
                 init_filter_reg=1e-2, min_filter_reg=1e-3, detach_length=float('Inf'), alpha_eps=0.0):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.hinge_threshold = hinge_threshold
        self.gauss_sigma = gauss_sigma
        self.alpha_eps = alpha_eps

    def get_label(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, -1, 1).to(center.device)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 1, -1).to(center.device)
        g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k0 - center[:,:,0].reshape(*center.shape[:2], 1, 1)) ** 2)
        g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k1 - center[:,:,1].reshape(*center.shape[:2], 1, 1)) ** 2)
        gauss = g0 * g1
        return gauss


    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1,)) - dmap_offset
        label_map = self.get_label(center, output_sz)
        target_mask = (label_map > self.hinge_threshold).float()
        label_map *= target_mask

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1)

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
            score_mask = target_mask + (1.0 - target_mask) * (scores.detach() > 0).float()
            residuals = sample_weight * (scores_act - label_map)

            if compute_losses:
                losses.append(((residuals**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

            # Compute gradient
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + \
                          reg_weight * weights

            # Map the gradient with the Jacobian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).sum(dim=(1,2,3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (step_length_factor * alpha.reshape(-1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
            losses.append((((sample_weight * (scores - label_map))**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

        return weights, weight_iterates, losses


class PrDiMPSteepestDescentNewton(nn.Module):
    """Optimizer module for PrDiMP.
    It unrolls the steepest descent with Newton iterations to optimize the target filter. See the PrDiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        gauss_sigma:  The standard deviation to use for the label density function.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
        init_uni_weight:  Weight of uniform label distribution.
        normalize_label:  Wheter to normalize the label distribution.
        label_shrink:  How much to shrink to label distribution.
        softmax_reg:  Regularization in the denominator of the SoftMax.
        label_threshold:  Threshold probabilities smaller than this.
    """
    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, gauss_sigma=1.0, min_filter_reg=1e-3, detach_length=float('Inf'),
                 alpha_eps=0.0, init_uni_weight=None, normalize_label=False, label_shrink=0, softmax_reg=None, label_threshold=0.0):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.uni_weight = 0 if init_uni_weight is None else init_uni_weight
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold

    def get_label_density(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, -1, 1).to(center.device)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 1, -1).to(center.device)
        dist0 = (k0 - center[:,:,0].reshape(*center.shape[:2], 1, 1)) ** 2
        dist1 = (k1 - center[:,:,1].reshape(*center.shape[:2], 1, 1)) ** 2
        if self.gauss_sigma == 0:
            dist0_view = dist0.reshape(-1, dist0.shape[-2])
            dist1_view = dist1.reshape(-1, dist1.shape[-1])
            one_hot0 = torch.zeros_like(dist0_view)
            one_hot1 = torch.zeros_like(dist1_view)
            one_hot0[torch.arange(one_hot0.shape[0]), dist0_view.argmin(dim=-1)] = 1.0
            one_hot1[torch.arange(one_hot1.shape[0]), dist1_view.argmin(dim=-1)] = 1.0
            gauss = one_hot0.reshape(dist0.shape) * one_hot1.reshape(dist1.shape)
        else:
            g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist0)
            g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist1)
            gauss = (g0 / (2*math.pi*self.gauss_sigma**2)) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= (gauss.sum(dim=(-2,-1), keepdim=True) + 1e-8)
        label_dens = (1.0 - self.label_shrink)*((1.0 - self.uni_weight) * gauss + self.uni_weight / (output_sz[0]*output_sz[1]))
        return label_dens

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute label density
        offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1,)) - offset
        label_density = self.get_label_density(center, output_sz)

        # Get total sample weights
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images]).to(feat.device)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.reshape(num_images, num_sequences, 1, 1)

        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)
        def _compute_loss(scores, weights):
            return torch.sum(sample_weight.reshape(sample_weight.shape[0], -1) *
                             (torch.log(scores.exp().sum(dim=(-2, -1)) + exp_reg) - (label_density * scores).sum(dim=(-2, -1)))) / num_sequences +\
                   reg_weight * (weights ** 2).sum() / num_sequences

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute "residuals"
            scores = filter_layer.apply_filter(feat, weights)
            scores_softmax = activation.softmax_reg(scores.reshape(num_images, num_sequences, -1), dim=2, reg=self.softmax_reg).reshape(scores.shape)
            res = sample_weight*(scores_softmax - label_density)

            if compute_losses:
                losses.append(_compute_loss(scores, weights))

            # Compute gradient
            weights_grad = filter_layer.apply_feat_transpose(feat, res, filter_sz, training=self.training) + \
                          reg_weight * weights

            # Map the gradient with the Hessian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(sm_scores_grad, dim=(-2,-1), keepdim=True)
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(num_images, num_sequences, -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (sample_weight.reshape(sample_weight.shape[0], -1) * grad_hes_grad).sum(dim=0)

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).sum(dim=(1,2,3))
            alpha_den = (grad_hes_grad + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (step_length_factor * alpha.reshape(-1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append(_compute_loss(scores, weights))

        return weights, weight_iterates, losses

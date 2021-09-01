#
# --------------------------------------------------------------------*/
# This file includes code from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
# --------------------------------------------------------------------*/
#


import torch
import torch.utils.checkpoint

from torch import nn

from copy import deepcopy, copy
from abc import ABCMeta, abstractmethod


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.
        required_data_keys: list of expected keys in the input data dictionary.
        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.
        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.
        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.
        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.
        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """
    base_default_conf = {
        'name': None,
        'trainable': True,  # if false: do not optimize this model parameters
        'freeze_batch_normalization': False,  # use test-time statistics
    }
    default_conf = {}
    # required_data_keys = []
    # strict_conf = True

    def __init__(self, conf=None):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = {}
        self.conf.update(**self.base_default_conf)
        self.conf.update(**self.default_conf)
        if conf is not None:
            self.conf.update(**conf)

        self.required_data_keys = copy(self.required_data_keys)

        if not self.conf['trainable']:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if self.conf['freeze_batch_normalization']:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim*self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim*2, num_dim*2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type):
        super().__init__()
        self.update = AttentionalPropagation(feature_dim, 4)
        assert layer_type in ['cross', 'self']
        self.type = layer_type

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError(self.type)
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_types, checkpointed=False):
        super().__init__()
        self.checkpointed = checkpointed
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type) for layer_type in layer_types])

    def forward(self, desc0, desc1):
        for layer in self.layers:
            if self.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                        layer, desc0, desc1, preserve_rng_state=False)
            else:
                desc0, desc1 = layer(desc0, desc1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(BaseModel):
    default_conf = {
        'input_dim': 256,
        'descriptor_dim': 256,
        'bottleneck_dim': None,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'output_normalization': 'sinkhorn',
        'num_sinkhorn_iterations': 50,
        'filter_threshold': 0.2,
        'checkpointed': False,
        'loss': {
            'nll_weight': 1.,
            'nll_balancing': 0.5,
            'reward_weight': 0.,
            'bottleneck_l2_weight': 0.,
        },
    }
    required_data_keys = [
            'img_coords0', 'img_coords1',
            'descriptors0', 'descriptors1',
            'scores0', 'scores1']

    def __init__(self, conf=None):
        super().__init__(conf=conf)
        if self.conf['bottleneck_dim'] is not None:
            self.bottleneck_down = nn.Conv1d(
                self.conf['input_dim'], self.conf['bottleneck_dim'],
                kernel_size=1, bias=True)
            self.bottleneck_up = nn.Conv1d(
                self.conf['bottleneck_dim'], self.conf['input_dim'],
                kernel_size=1, bias=True)
            nn.init.constant_(self.bottleneck_down.bias, 0.0)
            nn.init.constant_(self.bottleneck_up.bias, 0.0)

        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            self.input_proj = nn.Conv1d(
                self.conf['input_dim'], self.conf['descriptor_dim'],
                kernel_size=1, bias=True)
            nn.init.constant_(self.input_proj.bias, 0.0)

        self.kenc = KeypointEncoder(self.conf['descriptor_dim'], self.conf['keypoint_encoder'])

        if not self.conf['skip_gnn']:
            self.gnn = AttentionalGNN(
                self.conf['descriptor_dim'], self.conf['GNN_layers'], self.conf['checkpointed'])

        self.final_proj = nn.Conv1d(
            self.conf['descriptor_dim'], self.conf['descriptor_dim'], kernel_size=1, bias=True)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)

        bin_score = torch.nn.Parameter(torch.tensor(0.0))
        self.register_parameter('bin_score', bin_score)

    def _forward(self, data):
        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['img_coords0'], data['img_coords1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'match_scores0': kpts0.new_zeros(shape0),
                'match_scores1': kpts1.new_zeros(shape1),
            }

        if self.conf['bottleneck_dim'] is not None:
            pred['down_descriptors0'] = desc0 = self.bottleneck_down(desc0)
            pred['down_descriptors1'] = desc1 = self.bottleneck_down(desc1)
            desc0 = self.bottleneck_up(desc0)
            desc1 = self.bottleneck_up(desc1)
            desc0 = nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = nn.functional.normalize(desc1, p=2, dim=1)
            pred['bottleneck_descriptors0'] = desc0
            pred['bottleneck_descriptors1'] = desc1
            if self.conf['loss']['nll_weight'] == 0:
                desc0 = desc0.detach()
                desc1 = desc1.detach()

        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, data['image_size0'])
        kpts1 = normalize_keypoints(kpts1, data['image_size1'])

        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        if not self.conf['skip_gnn']:
            desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.conf['descriptor_dim']**.5

        if self.conf['output_normalization'] == 'sinkhorn':
            scores = log_optimal_transport(scores, self.bin_score, iters=self.conf['num_sinkhorn_iterations'])
        elif self.conf['output_normalization'] == 'double_softmax':
            scores = log_double_softmax(scores, self.bin_score)
        else:
            raise ValueError(self.conf['output_normalization'])

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf['filter_threshold'])
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))

        return {
            **pred,
            'log_assignment': scores,
            'matches0': m0,
            'matches1': m1,
            'match_scores0': mscores0,
            'match_scores1': mscores1,
            'bin_score': self.bin_score
        }

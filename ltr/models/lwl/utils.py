import torch
import torch.nn.functional as F


def adaptive_cat(seq, dim=0, ref_tensor=0, mode='bilinear'):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz, mode=mode) for t in seq], dim=dim)
    return t


def interpolate(t, sz, mode='bilinear'):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    align = {} if mode == 'nearest' else dict(align_corners=False)
    return F.interpolate(t, sz, mode=mode, **align) if t.shape[-2:] != sz else t


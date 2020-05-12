import torch
import math
from pytracking import fourier
from pytracking import complex
import torch.nn.functional as F


def hann1d(sz: int, centered = True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
    return torch.cat([w, w[1:sz-sz//2].flip((0,))])


def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)


def hann2d_clipped(sz: torch.Tensor, effective_sz: torch.Tensor, centered = True) -> torch.Tensor:
    """1D clipped cosine window."""

    # Ensure that the difference is even
    effective_sz += (effective_sz - sz) % 2
    effective_window = hann1d(effective_sz[0].item(), True).reshape(1, 1, -1, 1) * hann1d(effective_sz[1].item(), True).reshape(1, 1, 1, -1)

    pad = (sz - effective_sz) / 2

    window = F.pad(effective_window, (pad[1].item(), pad[1].item(), pad[0].item(), pad[0].item()), 'replicate')

    if centered:
        return window
    else:
        mid = (sz / 2).int()
        window_shift_lr = torch.cat((window[:, :, :, mid[1]:], window[:, :, :, :mid[1]]), 3)
        return torch.cat((window_shift_lr[:, :, mid[0]:, :], window_shift_lr[:, :, :mid[0], :]), 2)


def gauss_fourier(sz: int, sigma: float, half: bool = False) -> torch.Tensor:
    if half:
        k = torch.arange(0, int(sz/2+1))
    else:
        k = torch.arange(-int((sz-1)/2), int(sz/2+1))
    return (math.sqrt(2*math.pi) * sigma / sz) * torch.exp(-2 * (math.pi * sigma * k.float() / sz)**2)


def gauss_spatial(sz, sigma, center=0, end_pad=0):
    k = torch.arange(-(sz-1)/2, (sz+1)/2+end_pad)
    return torch.exp(-1.0/(2*sigma**2) * (k - center)**2)


def label_function(sz: torch.Tensor, sigma: torch.Tensor):
    return gauss_fourier(sz[0].item(), sigma[0].item()).reshape(1, 1, -1, 1) * gauss_fourier(sz[1].item(), sigma[1].item(), True).reshape(1, 1, 1, -1)

def label_function_spatial(sz: torch.Tensor, sigma: torch.Tensor, center: torch.Tensor = torch.zeros(2), end_pad: torch.Tensor = torch.zeros(2)):
    """The origin is in the middle of the image."""
    return gauss_spatial(sz[0].item(), sigma[0].item(), center[0], end_pad[0].item()).reshape(1, 1, -1, 1) * \
           gauss_spatial(sz[1].item(), sigma[1].item(), center[1], end_pad[1].item()).reshape(1, 1, 1, -1)


def cubic_spline_fourier(f, a):
    """The continuous Fourier transform of a cubic spline kernel."""

    bf = (6*(1 - torch.cos(2 * math.pi * f)) + 3*a*(1 - torch.cos(4 * math.pi * f))
           - (6 + 8*a)*math.pi*f*torch.sin(2 * math.pi * f) - 2*a*math.pi*f*torch.sin(4 * math.pi * f)) \
         / (4 * math.pi**4 * f**4)

    bf[f == 0] = 1

    return bf


def get_interp_fourier(sz: torch.Tensor, method='ideal', bicubic_param=0.5, centering=True, windowing=False, device='cpu'):

    ky, kx = fourier.get_frequency_coord(sz)

    if method=='ideal':
        interp_y = torch.ones(ky.shape) / sz[0]
        interp_x = torch.ones(kx.shape) / sz[1]
    elif method=='bicubic':
        interp_y = cubic_spline_fourier(ky / sz[0], bicubic_param) / sz[0]
        interp_x = cubic_spline_fourier(kx / sz[1], bicubic_param) / sz[1]
    else:
        raise ValueError('Unknown method.')

    if centering:
        interp_y = complex.mult(interp_y, complex.exp_imag((-math.pi/sz[0]) * ky))
        interp_x = complex.mult(interp_x, complex.exp_imag((-math.pi/sz[1]) * kx))

    if windowing:
        raise NotImplementedError

    return interp_y.to(device), interp_x.to(device)


def interpolate_dft(a: torch.Tensor, interp_fs) -> torch.Tensor:

    if isinstance(interp_fs, torch.Tensor):
        return complex.mult(a, interp_fs)
    if isinstance(interp_fs, (tuple, list)):
        return complex.mult(complex.mult(a, interp_fs[0]), interp_fs[1])
    raise ValueError('"interp_fs" must be tensor or tuple of tensors.')


def get_reg_filter(sz: torch.Tensor, target_sz: torch.Tensor, params):
    """Computes regularization filter in CCOT and ECO."""

    if not params.use_reg_window:
        return params.reg_window_min * torch.ones(1,1,1,1)

    if getattr(params, 'reg_window_square', False):
        target_sz = target_sz.prod().sqrt() * torch.ones(2)

    # Normalization factor
    reg_scale = 0.5 * target_sz

    # Construct grid
    if getattr(params, 'reg_window_centered', True):
        wrg = torch.arange(-int((sz[0]-1)/2), int(sz[0]/2+1), dtype=torch.float32).view(1,1,-1,1)
        wcg = torch.arange(-int((sz[1]-1)/2), int(sz[1]/2+1), dtype=torch.float32).view(1,1,1,-1)
    else:
        wrg = torch.cat([torch.arange(0, int(sz[0]/2+1), dtype=torch.float32),
                         torch.arange(-int((sz[0] - 1) / 2), 0, dtype=torch.float32)]).view(1,1,-1,1)
        wcg = torch.cat([torch.arange(0, int(sz[1]/2+1), dtype=torch.float32),
                         torch.arange(-int((sz[1] - 1) / 2), 0, dtype=torch.float32)]).view(1,1,1,-1)

    # Construct regularization window
    reg_window = (params.reg_window_edge - params.reg_window_min) * \
                 (torch.abs(wrg/reg_scale[0])**params.reg_window_power +
                  torch.abs(wcg/reg_scale[1])**params.reg_window_power) + params.reg_window_min

    # Compute DFT and enforce sparsity
    reg_window_dft = torch.rfft(reg_window, 2) / sz.prod()
    reg_window_dft_abs = complex.abs(reg_window_dft)
    reg_window_dft[reg_window_dft_abs < params.reg_sparsity_threshold * reg_window_dft_abs.max(), :] = 0

    # Do the inverse transform to correct for the window minimum
    reg_window_sparse = torch.irfft(reg_window_dft, 2, signal_sizes=sz.long().tolist())
    reg_window_dft[0,0,0,0,0] += params.reg_window_min - sz.prod() * reg_window_sparse.min()
    reg_window_dft = complex.real(fourier.rfftshift2(reg_window_dft))

    # Remove zeros
    max_inds,_ = reg_window_dft.nonzero().max(dim=0)
    mid_ind = int((reg_window_dft.shape[2]-1)/2)
    top = max_inds[-2].item() + 1
    bottom = 2*mid_ind - max_inds[-2].item()
    right = max_inds[-1].item() + 1
    reg_window_dft = reg_window_dft[..., bottom:top, :right]
    if reg_window_dft.shape[-1] > 1:
        reg_window_dft = torch.cat([reg_window_dft[..., 1:].flip((2, 3)), reg_window_dft], -1)

    return reg_window_dft


def max2d(a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax

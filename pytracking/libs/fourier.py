import torch
import torch.nn.functional as F
from pytracking import complex, TensorList
from pytracking.libs.tensorlist import tensor_operation


@tensor_operation
def rfftshift2(a: torch.Tensor):
    h = a.shape[2] + 2
    return torch.cat((a[:,:,(h-1)//2:,...], a[:,:,:h//2,...]), 2)


@tensor_operation
def irfftshift2(a: torch.Tensor):
    mid = int((a.shape[2]-1)/2)
    return torch.cat((a[:,:,mid:,...], a[:,:,:mid,...]), 2)


@tensor_operation
def cfft2(a):
    """Do FFT and center the low frequency component.
    Always produces odd (full) output sizes."""

    return rfftshift2(torch.rfft(a, 2))


@tensor_operation
def cifft2(a, signal_sizes=None):
    """Do inverse FFT corresponding to cfft2."""

    return torch.irfft(irfftshift2(a), 2, signal_sizes=signal_sizes)


@tensor_operation
def sample_fs(a: torch.Tensor, grid_sz: torch.Tensor = None, rescale = True):
    """Samples the Fourier series."""

    # Size of the fourier series
    sz = torch.Tensor([a.shape[2], 2*a.shape[3]-1]).float()

    # Default grid
    if grid_sz is None or sz[0] == grid_sz[0] and sz[1] == grid_sz[1]:
        if rescale:
            return sz.prod().item() * cifft2(a)
        return cifft2(a)

    if sz[0] > grid_sz[0] or sz[1] > grid_sz[1]:
        raise ValueError("Only grid sizes that are smaller than the Fourier series size are supported.")

    tot_pad = (grid_sz - sz).tolist()
    is_even = [s.item() % 2 == 0 for s in sz]

    # Compute paddings
    pad_top = int((tot_pad[0]+1)/2) if is_even[0] else int(tot_pad[0]/2)
    pad_bottom = int(tot_pad[0] - pad_top)
    pad_right = int((tot_pad[1]+1)/2)

    if rescale:
        return grid_sz.prod().item() * cifft2(F.pad(a, (0, 0, 0, pad_right, pad_top, pad_bottom)), signal_sizes=grid_sz.long().tolist())
    else:
        return cifft2(F.pad(a, (0, 0, 0, pad_right, pad_top, pad_bottom)), signal_sizes=grid_sz.long().tolist())


def get_frequency_coord(sz, add_complex_dim = False, device='cpu'):
    """Frequency coordinates."""

    ky = torch.arange(-int((sz[0]-1)/2), int(sz[0]/2+1), dtype=torch.float32, device=device).view(1,1,-1,1)
    kx = torch.arange(0, int(sz[1]/2+1), dtype=torch.float32, device=device).view(1,1,1,-1)

    if add_complex_dim:
        ky = ky.unsqueeze(-1)
        kx = kx.unsqueeze(-1)

    return ky, kx


@tensor_operation
def shift_fs(a: torch.Tensor, shift: torch.Tensor):
    """Shift a sample a in the Fourier domain.
    Params:
        a : The fourier coefficiens of the sample.
        shift : The shift to be performed normalized to the range [-pi, pi]."""

    if a.dim() != 5:
        raise ValueError('a must be the Fourier coefficients, a 5-dimensional tensor.')

    if shift[0] == 0 and shift[1] == 0:
        return a

    ky, kx = get_frequency_coord((a.shape[2], 2*a.shape[3]-1), device=a.device)

    return complex.mult(complex.mult(a, complex.exp_imag(shift[0].item()*ky)), complex.exp_imag(shift[1].item()*kx))


def sum_fs(a: TensorList) -> torch.Tensor:
    """Sum a list of Fourier series expansions."""

    s = None
    mid = None

    for e in sorted(a, key=lambda elem: elem.shape[-3], reverse=True):
        if s is None:
            s = e.clone()
            mid = int((s.shape[-3] - 1) / 2)
        else:
            # Compute coordinates
            top = mid - int((e.shape[-3] - 1) / 2)
            bottom = mid + int(e.shape[-3] / 2) + 1
            right = e.shape[-2]

            # Add the data
            s[..., top:bottom, :right, :] += e

    return s


def sum_fs12(a: TensorList) -> torch.Tensor:
    """Sum a list of Fourier series expansions."""

    s = None
    mid = None

    for e in sorted(a, key=lambda elem: elem.shape[0], reverse=True):
        if s is None:
            s = e.clone()
            mid = int((s.shape[0] - 1) / 2)
        else:
            # Compute coordinates
            top = mid - int((e.shape[0] - 1) / 2)
            bottom = mid + int(e.shape[0] / 2) + 1
            right = e.shape[1]

            # Add the data
            s[top:bottom, :right, ...] += e

    return s


@tensor_operation
def inner_prod_fs(a: torch.Tensor, b: torch.Tensor):
    if complex.is_complex(a) and complex.is_complex(b):
        return 2 * (a.reshape(-1) @ b.reshape(-1)) - a[:, :, :, 0, :].reshape(-1) @ b[:, :, :, 0, :].reshape(-1)
    elif complex.is_real(a) and complex.is_real(b):
        return 2 * (a.reshape(-1) @ b.reshape(-1)) - a[:, :, :, 0].reshape(-1) @ b[:, :, :, 0].reshape(-1)
    else:
        raise NotImplementedError('Not implemented for mixed real and complex.')
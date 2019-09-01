import torch
import torch.nn.functional as F
import numpy as np


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def sample_patch_transformed(im, pos, scale, image_sz, transforms):
    """Extract transformed image samples.
    args:
        im: Image.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche
    im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz)

    # Apply transforms
    im_patches = torch.cat([T(im_patch) for T in transforms])

    return im_patches


def sample_patch_multiscale(im, pos, scales, image_sz, mode: str='replicate'):
    """Extract image patches at multiple scales.
    args:
        im: Image.
        pos: Center position for extraction.
        scales: Image scales to extract image patches from.
        image_sz: Size to resize the image samples to
        mode: how to treat image borders: 'replicate' (default) or 'inside'
    """
    if isinstance(scales, (int, float)):
        scales = [scales]

    # Get image patches
    patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=mode) for s in scales))
    im_patches = torch.cat(list(patch_iter))
    patch_coords = torch.cat(list(coord_iter))

    return  im_patches, patch_coords


def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate'):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default) or 'inside'
    """

    if mode not in ['replicate', 'inside']:
        raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    # Get new sample size if forced inside the image
    if mode == 'inside':
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz).max().clamp(1)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) / df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1)/2
    br = posl + szl/2 + 1

    # Shift the crop to inside
    if mode == 'inside':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        # Get image patch
        im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]
    else:
        # Get image patch
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), mode)

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')

    return im_patch, patch_coord

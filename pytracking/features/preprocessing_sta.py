import torch
import torch.nn.functional as F
import numpy as np


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()

def sample_patch_transformed(im, bbox, pos, scale, image_sz, transforms, is_mask=False):
    """Extract transformed image samples.
    args:
        im: Image.
        bbox: Bounding box.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche
    im_patch, _, bbox_patch = sample_patch(im, bbox, pos, scale*image_sz, image_sz, is_mask=is_mask)

    # Apply transforms
    im_patches = torch.cat([T(im_patch, is_mask=is_mask) for T in transforms])

    return im_patches, bbox_patch


def sample_patch_multiscale(im, bbox, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
    """Extract image patches at multiple scales.
    args:
        im: Image.
        pos: Center position for extraction.
        scales: Image scales to extract image patches from.
        image_sz: Size to resize the image samples to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """
    if isinstance(scales, (int, float)):
        scales = [scales]

    # Get image patches
    patch_iter, coord_iter, bbox_iter = zip(*(sample_patch(im, bbox, pos, s*image_sz, image_sz, mode=mode,
                                                max_scale_change=max_scale_change) for s in scales))
    im_patches = torch.cat(list(patch_iter))
    patch_coords = torch.cat(list(coord_iter))
    patch_bboxes = torch.cat(list(bbox_iter))

    return  im_patches, patch_coords, patch_bboxes


def sample_patch(im: torch.Tensor, bbox: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        bbox: Bounding box.
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    bbox_clone = bbox.clone()

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
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
        bbox_clone[..., :2] -= os
        bbox_clone /= df
    else:
        im2 = im
    # plot_image(torch_to_numpy(im2), bbox_clone, "out_mid.jpg")
    # exit()
    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1)/2
    br = posl + szl/2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]
 
    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))
    bbox_clone[:, :, 0] -= tl[1].item()
    bbox_clone[:, :, 1] -= tl[0].item()
    
    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord, bbox_clone

    bbox_clone[:, :, :2] /= (torch.tensor(im_patch.shape[-2:]) / output_sz)
    bbox_clone[:, :, 2:] /= (torch.tensor(im_patch.shape[-2:]) / output_sz)
    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')
    

    return im_patch, patch_coord, bbox_clone


def crop_and_resize(im, crop_bb, output_sz, mask=None):
    x1 = crop_bb[0]
    x2 = crop_bb[0] + crop_bb[2]

    y1 = crop_bb[1]
    y2 = crop_bb[1] + crop_bb[3]

    # Crop target
    im_crop = F.pad(im, (-x1, x2 - im.shape[-1], -y1, y2 - im.shape[-2]), 'replicate')
    im_crop = F.interpolate(im_crop, output_sz.long().tolist(), mode='bilinear')

    if mask is None:
        return im_crop

    mask_crop = F.pad(mask, (-x1, x2 - im.shape[-1], -y1, y2 - im.shape[-2]))
    mask_crop = F.interpolate(mask_crop, output_sz.long().tolist(), mode='nearest')

    return im_crop, mask_crop

def plot_image(image, bbox=None, savePath="out.png"):
    from PIL import Image, ImageDraw
    ##image shape: h, w, 3
    im = Image.fromarray(image.astype(np.uint8))
    drawer = ImageDraw.Draw(im)
    ##bbox shape: 1, 1, 4
    shape = bbox.round().squeeze(0).squeeze(0).numpy().astype(np.int)
    shape[2:] += shape[:2]
    drawer.rectangle([(shape[0],shape[1]),(shape[2], shape[3])], outline="red") 
    im.save(savePath)
    print("image saved")


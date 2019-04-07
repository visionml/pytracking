import torch
import math
import cv2 as cv
import random
from random import gauss


def stack_tensors(x):
    if isinstance(x, list) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


def sample_target(im, target_bb, search_area_factor, output_sz=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    x, y, w, h = target_bb.tolist()

    # Crop image
    crop_sz = math.ceil(math.sqrt(w*h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - crop_sz*0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        return cv.resize(im_crop_padded, (output_sz, output_sz)), resize_factor
    else:
        return im_crop_padded, 1.0


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5*box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5*box_in[2:4]

    box_out_center = (crop_sz-1)/2 + (box_in_center - box_extract_center)*resize_factor
    box_out_wh = box_in[2:4]*resize_factor

    box_out = torch.cat((box_out_center - 0.5*box_out_wh, box_out_wh))
    return box_out


def centered_crop(frames, anno, area_factor, output_sz):
    crops_resize_factors = [sample_target(f, a, area_factor, output_sz)
                            for f, a in zip(frames, anno)]

    frames_crop, resize_factors = zip(*crops_resize_factors)

    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    anno_crop = [transform_image_to_crop(a, a, rf, crop_sz)
                 for a, rf in zip(anno, resize_factors)]

    return frames_crop, anno_crop


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized
        
    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """
    crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                            for f, a in zip(frames, box_extract)]

    frames_crop, resize_factors = zip(*crops_resize_factors)

    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                 for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    return frames_crop, box_crop


def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:,:2], proposals[:,:2])
    br = torch.min(reference[:,:2] + reference[:,2:], proposals[:,:2] + proposals[:,2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:,2:].prod(dim=1) + proposals[:,2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2]*box[3])*c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5*box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = gauss(c_x, perturb_factor[0])
        c_y_per = gauss(c_y, perturb_factor[1])

        w_per = gauss(box[2], perturb_factor[2])
        h_per = gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2]*rand_uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3]*rand_uniform(0.15, 0.5)

        box_per = torch.Tensor([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2]*rand_uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3]*rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou

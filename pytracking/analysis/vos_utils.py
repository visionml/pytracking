import warnings
import numpy as np
from skimage.morphology import binary_dilation, disk
from math import floor


def text_bargraph(values):

    blocks = np.array(('u', ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', 'o'))
    nsteps = len(blocks)-2-1
    hstep = 1 / (2*nsteps)
    values = np.array(values)
    nans = np.isnan(values)
    values[nans] = 0  # '░'
    indices = ((values + hstep) * nsteps + 1).astype(np.int)
    indices[values < 0] = 0
    indices[values > 1] = len(blocks)-1
    graph = blocks[indices]
    graph[nans] = '░'
    graph = str.join('', graph)
    return graph


# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
# -----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

# Originally db_eval_iou() in the davis challenge toolkit:
def davis_jaccard_measure(fg_mask, gt_mask):
    """ Compute region similarity as the Jaccard Index.

    :param fg_mask: (ndarray): binary segmentation map.
    :param gt_mask: (ndarray): binary annotation map.
    :return: jaccard (float): region similarity
    """

    gt_mask = gt_mask.astype(np.bool)
    fg_mask = fg_mask.astype(np.bool)

    if np.isclose(np.sum(gt_mask), 0) and np.isclose(np.sum(fg_mask), 0):
        return 1
    else:
        return np.sum((gt_mask & fg_mask)) / \
               np.sum((gt_mask | fg_mask), dtype=np.float32)


def davis_jaccard_measure_torch(fg_mask, gt_mask):
    """ Compute region similarity as the Jaccard Index.

    :param fg_mask: (ndarray): binary segmentation map.
    :param gt_mask: (ndarray): binary annotation map.
    :return: jaccard (float): region similarity
    """

    #gt_mask = gt_mask.astype(np.bool)
    #fg_mask = fg_mask.astype(np.bool)

    if gt_mask.sum() == 0 and fg_mask.sum() == 0:
        return 1
    else:
        return (gt_mask & fg_mask).sum() / \
               (gt_mask | fg_mask).sum().float()

# Originally db_eval_boundary() in the davis challenge toolkit:
def davis_f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def nanmean(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(*args, **kwargs)


def mean(X):
    """
    Compute average ignoring NaN values.
    """
    return np.nanmean(X)


def recall(X, threshold=0.5):
    """
    Fraction of values of X scoring higher than 'threshold'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = X[~np.isnan(X)]
        x = mean(x > threshold)
    return x


def decay(X, n_bins=4):
    """
    Performance loss over time.
    """
    X = X[~np.isnan(X)]
    ids = np.round(np.linspace(1, len(X), n_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [X[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])
    return D


def std(X):
    """
    Compute standard deviation.
    """
    return np.nanstd(X)



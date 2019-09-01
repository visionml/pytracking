import torch
import torch.nn.functional as F


def apply_filter(feat, filter):
    """Applies the filter on the input features (feat).
    args:
        feat: These are the input features. Must have dimensions (images_in_sequence, sequences, feat_dim, H, W)
        filter: The filter to apply. Must have dimensions (sequences, feat_dim, fH, fW) or (sequences, filters, feat_dim, fH, fW)
    output:
        scores: Output of filtering. Dimensions (images_in_sequence, sequences, yH, yW) or (images_in_sequence, sequences, filters, yH, yW)
    """

    multiple_filters = (filter.dim() == 5)

    padding = (filter.shape[-2] // 2, filter.shape[-1] // 2)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1

    if multiple_filters:
        scores = F.conv2d(feat.view(num_images, -1, feat.shape[-2], feat.shape[-1]), filter.view(-1, *filter.shape[-3:]),
                          padding=padding, groups=num_sequences)

        return scores.view(num_images, num_sequences, -1, scores.shape[-2], scores.shape[-1])

    scores = F.conv2d(feat.view(num_images, -1, feat.shape[-2], feat.shape[-1]), filter,
                      padding=padding, groups=num_sequences)

    return scores.view(num_images, num_sequences, scores.shape[-2], scores.shape[-1])


def apply_feat_transpose(feat, input, filter_ksz, training=True):
    """Applies the transposed operation off apply_filter w.r.t. filter itself. Can be used to compute the filter gradient.
    args:
        feat: These are the input features. Must have dimensions (images_in_sequence, sequences, feat_dim, H, W)
        input: Input activation (e.g. residuals). Must have dimensions (images_in_sequence, sequences, yH, yW) or
                (images_in_sequence, sequences, filters, yH, yW)
        training: Choose the faster implementation whether training or not.
    output:
        Output of transposed operation. Dimensions (sequences, feat_dim, fH, fW)
    """

    if training or input.dim() == 5:
        return _apply_feat_transpose_v3(feat, input, filter_ksz)
    return _apply_feat_transpose_v2(feat, input, filter_ksz)


def _apply_feat_transpose_v1(feat, input, filter_ksz):
    """This one is slow as hell!!!!"""

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    feat_sz = (feat.shape[-2], feat.shape[-1])
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    # trans_pad = sz + padding - filter_ksz
    trans_pad = [sz + ksz//2 - ksz for sz, ksz in zip(feat_sz, filter_ksz)]

    filter_grad = F.conv_transpose2d(input.flip((2, 3)).view(1, -1, input.shape[-2], input.shape[-1]),
                                     feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]),
                                     padding=trans_pad, groups=num_images * num_sequences)

    return filter_grad.view(num_images, num_sequences, -1, filter_grad.shape[-2], filter_grad.shape[-1]).sum(dim=0)


def _apply_feat_transpose_v2(feat, input, filter_ksz):
    """Fast forward and slow backward"""

    multiple_filters = (input.dim() == 5)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    num_filters = input.shape[2] if multiple_filters else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [(ksz-1)//2 for ksz in filter_ksz]

    if multiple_filters:
        filter_grad = F.conv2d(input.view(-1, num_filters, input.shape[-2], input.shape[-1]).permute(1,0,2,3),
                               feat.view(-1, 1, feat.shape[-2], feat.shape[-1]),
                               padding=trans_pad, groups=num_images * num_sequences)

        if num_images == 1:
            return filter_grad.view(num_filters, num_sequences, -1, filter_grad.shape[-2], filter_grad.shape[-1]).flip((3,4)).permute(1,0,2,3,4)
        return filter_grad.view(num_filters, num_images, num_sequences, -1, filter_grad.shape[-2], filter_grad.shape[-1]).sum(dim=1).flip((3,4)).permute(1,0,2,3,4)

    filter_grad = F.conv2d(input.view(1, -1, input.shape[-2], input.shape[-1]),
                                     feat.view(-1, 1, feat.shape[-2], feat.shape[-1]),
                                     padding=trans_pad, groups=num_images * num_sequences)

    return filter_grad.view(num_images, num_sequences, -1, filter_grad.shape[-2], filter_grad.shape[-1]).sum(dim=0).flip((2,3))


def _apply_feat_transpose_v3(feat, input, filter_ksz):
    """Slow forward fast backward"""

    multiple_filters = (input.dim() == 5)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    num_filters = input.shape[2] if multiple_filters else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [ksz//2 for  ksz in filter_ksz]

    filter_grad = F.conv2d(feat.view(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]).permute(1,0,2,3),
                           input.view(-1, 1, input.shape[-2], input.shape[-1]),
                           padding=trans_pad, groups=num_images * num_sequences)

    if multiple_filters:
        if num_images == 1:
            return filter_grad.view(-1, num_sequences, num_filters, filter_grad.shape[-2], filter_grad.shape[-1]).permute(1,2,0,3,4)
        return filter_grad.view(-1, num_images, num_sequences, num_filters, filter_grad.shape[-2], filter_grad.shape[-1]).sum(dim=1).permute(1,2,0,3,4)

    if num_images == 1:
        return filter_grad.permute(1,0,2,3)
    return filter_grad.view(-1, num_images, num_sequences, filter_grad.shape[-2], filter_grad.shape[-1]).sum(dim=1).permute(1,0,2,3)


def _apply_feat_transpose_v4(feat, input, filter_ksz):
    """Slow forward fast backward"""

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [ksz//2 for  ksz in filter_ksz]

    filter_grad = F.conv2d(feat.permute(2,1,0,3,4).reshape(feat.shape[-3], -1, feat.shape[-2], feat.shape[-1]),
                           input.permute(1,0,2,3),
                           padding=trans_pad, groups=num_sequences)

    return filter_grad.permute(1,0,2,3)



def filter_gradient(feat, filter, label=None, training=True):
    """Computes gradient of the filter when applied on the input features and ground truth label.
    args:
        feat: These are the input features. Must have dimensions (images_in_sequence, sequences, feat_dim, H, W)
        filter: The filter to apply. Must have dimensions (sequences, feat_dim, fH, fW)
        label: Ground truth label in the L2 loss. Dimensions (images_in_sequence, sequences, yH, yW)
    output:
        filter_gradient: Dimensions same as input filter (sequences, feat_dim, fH, fW)
    """

    residuals = apply_filter(feat, filter)
    if label is not None:
        residuals = residuals - label
    filter_ksz = (filter.shape[-2], filter.shape[-1])
    return apply_feat_transpose(feat, residuals, filter_ksz, training=training)

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import ltr.models.loss.lovasz_loss as lovasz_loss


class LovaszSegLoss(nn.Module):
    def __init__(self, classes=[1,], per_image=True):
        super().__init__()

        self.classes = classes
        self.per_image=per_image

    def forward(self, input, target):
        return lovasz_loss.lovasz_softmax(probas=torch.sigmoid(input), labels=target, per_image=self.per_image, classes=self.classes)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device = None,
            dtype = None,
            eps = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        #>>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        #>>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes, shape[1], shape[2])).to(device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

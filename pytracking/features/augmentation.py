import numpy as np
import math
import torch
import torch.nn.functional as F
import cv2 as cv
from pytracking.features.preprocessing import numpy_to_torch, torch_to_numpy


class Transform:
    """Base data augmentation transform class."""

    def __init__(self, output_sz = None, shift = None):
        self.output_sz = output_sz
        self.shift = (0,0) if shift is None else shift

    def __call__(self, image):
        raise NotImplementedError

    def crop_to_output(self, image):
        if isinstance(image, torch.Tensor):
            imsz = image.shape[2:]
            if self.output_sz is None:
                pad_h = 0
                pad_w = 0
            else:
                pad_h = (self.output_sz[0] - imsz[0]) / 2
                pad_w = (self.output_sz[1] - imsz[1]) / 2

            pad_left = math.floor(pad_w) + self.shift[1]
            pad_right = math.ceil(pad_w) - self.shift[1]
            pad_top = math.floor(pad_h) + self.shift[0]
            pad_bottom = math.ceil(pad_h) - self.shift[0]

            return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 'replicate')
        else:
            raise NotImplementedError

class Identity(Transform):
    """Identity transformation."""
    def __call__(self, image):
        return self.crop_to_output(image)

class FlipHorizontal(Transform):
    """Flip along horizontal axis."""
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((3,)))
        else:
            return np.fliplr(image)

class FlipVertical(Transform):
    """Flip along vertical axis."""
    def __call__(self, image: torch.Tensor):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((2,)))
        else:
            return np.flipud(image)

class Translation(Transform):
    """Translate."""
    def __init__(self, translation, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.shift = (self.shift[0] + translation[0], self.shift[1] + translation[1])

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image)
        else:
            raise NotImplementedError

class Scale(Transform):
    """Scale."""
    def __init__(self, scale_factor, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.scale_factor = scale_factor

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # Calculate new size. Ensure that it is even so that crop/pad becomes easier
            h_orig, w_orig = image.shape[2:]

            if h_orig != w_orig:
                raise NotImplementedError

            h_new = round(h_orig /self.scale_factor)
            h_new += (h_new - h_orig) % 2
            w_new = round(w_orig /self.scale_factor)
            w_new += (w_new - w_orig) % 2

            image_resized = F.interpolate(image, [h_new, w_new], mode='bilinear')

            return self.crop_to_output(image_resized)
        else:
            raise NotImplementedError


class Affine(Transform):
    """Affine transformation."""
    def __init__(self, transform_matrix, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.transform_matrix = transform_matrix

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(numpy_to_torch(self(torch_to_numpy(image))))
        else:
            return cv.warpAffine(image, self.transform_matrix, image.shape[1::-1], borderMode=cv.BORDER_REPLICATE)


class Rotate(Transform):
    """Rotate with given angle."""
    def __init__(self, angle, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.angle = math.pi * angle/180

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(numpy_to_torch(self(torch_to_numpy(image))))
        else:
            c = (np.expand_dims(np.array(image.shape[:2]),1)-1)/2
            R = np.array([[math.cos(self.angle), math.sin(self.angle)],
                          [-math.sin(self.angle), math.cos(self.angle)]])
            H =np.concatenate([R, c - R @ c], 1)
            return cv.warpAffine(image, H, image.shape[1::-1], borderMode=cv.BORDER_REPLICATE)


class Blur(Transform):
    """Blur with given sigma (can be axis dependent)."""
    def __init__(self, sigma, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            sz = image.shape[2:]
            im1 = F.conv2d(image.view(-1,1,sz[0],sz[1]), self.filter[0], padding=(self.filter_size[0],0))
            return self.crop_to_output(F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(1,-1,sz[0],sz[1]))
        else:
            raise NotImplementedError
import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)



class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords = coords.clone()
            coords[1,:] = (image_shape[1] - 1) - coords[1,:]
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask


class Blur(TransformBase):
    """ Blur the image by applying a gaussian kernel with given sigma"""
    def __init__(self, sigma):
        super().__init__()
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def transform_image(self, image):
        if torch.is_tensor(image):
            sz = image.shape[2:]
            im1 = F.conv2d(image.view(-1, 1, sz[0], sz[1]), self.filter[0], padding=(self.filter_size[0], 0))
            return F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(-1,sz[0],sz[1])
        else:
            raise NotImplementedError


class RandomBlur(TransformBase):
    """ Blur the image, with a given probability, by applying a gaussian kernel with given sigma"""
    def __init__(self, sigma, probability=0.1):
        super().__init__()
        self.probability = probability

        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def roll(self):
        return random.random() < self.probability

    def transform(self, image, do_blur=None):
        if do_blur is None:
            do_blur = False

        if do_blur:
            if torch.is_tensor(image):
                sz = image.shape[1:]
                im1 = F.conv2d(image.view(-1, 1, sz[0], sz[1]), self.filter[0], padding=(self.filter_size[0], 0))
                return F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(-1,sz[0],sz[1])
            else:
                raise NotImplementedError
        else:
            return image


class RandomAffine(TransformBase):
    """Apply random affine transformation."""
    def __init__(self, p_flip=0.0, max_rotation=0.0, max_shear=0.0, max_scale=0.0, max_ar_factor=0.0,
                 border_mode='constant', pad_amount=0):
        super().__init__()
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.max_ar_factor = max_ar_factor

        if border_mode == 'constant':
            self.border_flag = cv.BORDER_CONSTANT
        elif border_mode == 'replicate':
            self.border_flag == cv.BORDER_REPLICATE
        else:
            raise Exception

        self.pad_amount = pad_amount

    def roll(self):
        do_flip = random.random() < self.p_flip
        theta = random.uniform(-self.max_rotation, self.max_rotation)

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)

        ar_factor = np.exp(random.uniform(-self.max_ar_factor, self.max_ar_factor))
        scale_factor = np.exp(random.uniform(-self.max_scale, self.max_scale))

        return do_flip, theta, (shear_x, shear_y), (scale_factor, scale_factor * ar_factor)

    def _construct_t_mat(self, image_shape, do_flip, theta, shear_values, scale_factors):
        im_h, im_w = image_shape
        t_mat = np.identity(3)

        if do_flip:
            if do_flip:
                t_mat[0, 0] = -1.0
                t_mat[0, 2] = im_w

        t_rot = cv.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
        t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

        t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_scale = np.array([[scale_factors[0], 0.0, (1.0 - scale_factors[0]) * 0.5 * im_w],
                            [0.0, scale_factors[1], (1.0 - scale_factors[1]) * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_mat = t_scale @ t_rot @ t_shear @ t_mat

        t_mat[0, 2] += self.pad_amount
        t_mat[1, 2] += self.pad_amount

        t_mat = t_mat[:2, :]

        return t_mat

    def transform_image(self, image, do_flip, theta, shear_values, scale_factors):
        if torch.is_tensor(image):
            raise Exception('Only supported for numpy input')

        t_mat = self._construct_t_mat(image.shape[:2], do_flip, theta, shear_values, scale_factors)
        output_sz = (image.shape[1] + 2*self.pad_amount, image.shape[0] + 2*self.pad_amount)
        image_t = cv.warpAffine(image, t_mat, output_sz, flags=cv.INTER_LINEAR,
                                borderMode=self.border_flag)

        return image_t

    def transform_coords(self, coords, image_shape, do_flip, theta, shear_values, scale_factors):
        t_mat = self._construct_t_mat(image_shape, do_flip, theta, shear_values, scale_factors)

        t_mat_tensor = torch.from_numpy(t_mat).float()

        coords_xy1 = torch.stack((coords[1, :], coords[0, :], torch.ones_like(coords[1, :])))

        coords_xy_t = torch.mm(t_mat_tensor, coords_xy1)

        return coords_xy_t[[1, 0], :]

    def transform_mask(self, mask, do_flip, theta, shear_values, scale_factors):
        t_mat = self._construct_t_mat(mask.shape[:2], do_flip, theta, shear_values, scale_factors)
        output_sz = (mask.shape[1] + 2*self.pad_amount, mask.shape[0] + 2*self.pad_amount)

        mask_t = cv.warpAffine(mask.numpy(), t_mat, output_sz, flags=cv.INTER_NEAREST,
                               borderMode=self.border_flag)

        return torch.from_numpy(mask_t)

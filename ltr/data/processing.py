import torch
import math
import numpy as np
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ATOMProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposal_method = self.proposal_params.get('proposal_method', 'default')

        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)
            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                             num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class KLBBregProcessing(BaseProcessing):
    """ Based on ATOMProcessing. It supports training ATOM using the Maximum Likelihood or KL-divergence based learning
    introduced in [https://arxiv.org/abs/1909.12297] and in PrDiMP [https://arxiv.org/abs/2003.12565].
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params[
                                                                             'boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get(
                                                                             'add_mean_box', False))

        return proposals, proposal_density, gt_density

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_density', 'gt_density'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class ATOMwKLProcessing(BaseProcessing):
    """Same as ATOMProcessing but using the GMM-based sampling of proposal boxes used in KLBBregProcessing."""
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         self.proposal_params['gt_sigma'],
                                                                         self.proposal_params['boxes_per_frame'])

        iou = prutils.iou_gen(proposals, box.view(1, 4))
        return proposals, proposal_density, gt_density, iou

    def __call__(self, data: TensorDict):
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density, proposal_iou = zip(
            *[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density
        data['proposal_iou'] = proposal_iou
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data



class DiMPProcessing(BaseProcessing):
    """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the test images. A set of proposals are
    also generated for the test images by jittering the ground truth box. These proposals are used to train the
    bounding box estimating branch.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
            max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposal_method = self.proposal_params.get('proposal_method', 'default')

        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)

            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                     num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1, 4), proposals.view(-1, 4))
        else:
            raise ValueError('Unknown proposal method.')

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_iou',
                'test_label' (optional), 'train_label' (optional), 'test_label_density' (optional), 'train_label_density' (optional)
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

            data['test_proposals'] = list(frame2_proposals)
            data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

        return data


class KLDiMPProcessing(BaseProcessing):
    """ The processing class used for training PrDiMP that additionally supports the probabilistic classifier and
    bounding box regressor. See DiMPProcessing for details.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', proposal_params=None,
                 label_function_params=None, label_density_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
            max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
            label_density_params - Arguments for the label density generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generate proposal sample boxes from a GMM proposal distribution and compute their ground-truth density.
        This is used for ML and KL based regression learning of the bounding box regressor.
        args:
            box - input bounding box
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params['boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_density', 'gt_density',
                'test_label' (optional), 'train_label' (optional), 'test_label_density' (optional), 'train_label_density' (optional)
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
        if self.label_density_params is not None:
            data['train_label_density'] = self._generate_label_density(data['train_anno'])
            data['test_label_density'] = self._generate_label_density(data['test_anno'])

        return data


class LWLProcessing(BaseProcessing):
    """ The processing class used for training LWL. The images are processed in the following way.
    First, the target bounding box (computed using the segmentation mask)is jittered by adding some noise.
    Next, a rectangular region (called search region ) centered at the jittered target center, and of area
    search_area_factor^2 times the area of the jittered box is cropped from the image.
    The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. The argument 'crop_type' determines how out-of-frame regions are handled when cropping the
    search region. For instance, if crop_type == 'replicate', the boundary pixels are replicated in case the search
    region crop goes out of frame. If crop_type == 'inside_major', the search region crop is shifted/shrunk to fit
    completely inside one axis of the image.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', new_roll=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - The size (width, height) to which the search region is resized. The aspect ratio is always
                        preserved when resizing the search region
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - Determines how out-of-frame regions are handled when cropping the search region.
                        If 'replicate', the boundary pixels are replicated in case the search region crop goes out of
                                        image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis
                        of the image.
            max_scale_change - Maximum allowed scale change when shrinking the search region to fit the image
                               (only applicable to 'inside' and 'inside_major' cropping modes). In case the desired
                               shrink factor exceeds the max_scale_change, the search region is only shrunk to the
                               factor max_scale_change. Out-of-frame regions are then handled by replicating the
                               boundary pixels. If max_scale_change is set to None, unbounded shrinking is allowed.

            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            new_roll - Whether to use the same random roll values for train and test frames when applying the joint
                       transformation. If True, a new random roll is performed for the test frame transformations. Thus,
                       if performing random flips, the set of train frames and the set of test frames will be flipped
                       independently.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            jittered_size = box[2:4] * torch.exp(torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                                                               self.scale_jitter_factor[mode]))
        else:
            raise Exception

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode])).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        # Apply joint transformations. i.e. All train/test frames in a sequence are applied the transformation with the
        # same parameters
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](
                image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](
                image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=self.new_roll)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            # Extract a crop containing the target
            crops, boxes, mask_crops = prutils.target_image_crop(data[s + '_images'], jittered_anno,
                                                                 data[s + '_anno'], self.search_area_factor,
                                                                 self.output_sz, mode=self.crop_type,
                                                                 max_scale_change=self.max_scale_change,
                                                                 masks=data[s + '_masks'])

            # Apply independent transformations to each image
            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=crops, bbox=boxes, mask=mask_crops, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class KYSProcessing(BaseProcessing):
    """ The processing class used for training KYS. The images are processed in the following way.
        First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
        centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
        cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
        always at the center of the search region. The search region is then resized to a fixed size given by the
        argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
        used for computing the loss of the predicted classification model on the test images. A set of proposals are
        also generated for the test images by jittering the ground truth box. These proposals can be used to train the
        bounding box estimating branch.
        """
    def __init__(self, search_area_factor, output_sz, center_jitter_param, scale_jitter_param,
                 proposal_params=None, label_function_params=None, min_crop_inside_ratio=0,
                 *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _generate_synthetic_motion for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _generate_synthetic_motion for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
            min_crop_inside_ratio - Minimum amount of cropped search area which should be inside the image.
                                    See _check_if_crop_inside_image for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_param = center_jitter_param
        self.scale_jitter_param = scale_jitter_param

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.min_crop_inside_ratio = min_crop_inside_ratio

    def _check_if_crop_inside_image(self, box, im_shape):
        x, y, w, h = box.tolist()

        if w <= 0.0 or h <= 0.0:
            return False

        crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)

        x1 = x + 0.5 * w - crop_sz * 0.5
        x2 = x1 + crop_sz

        y1 = y + 0.5 * h - crop_sz * 0.5
        y2 = y1 + crop_sz

        w_inside = max(min(x2, im_shape[1]) - max(x1, 0), 0)
        h_inside = max(min(y2, im_shape[0]) - max(y1, 0), 0)

        crop_area = ((x2 - x1) * (y2 - y1))

        if crop_area > 0:
            inside_ratio = w_inside * h_inside / crop_area
            return inside_ratio > self.min_crop_inside_ratio
        else:
            return False

    def _generate_synthetic_motion(self, boxes, images, mode):
        num_frames = len(boxes)

        out_boxes = []

        for i in range(num_frames):
            jittered_box = None
            for _ in range(10):
                orig_box = boxes[i]
                jittered_size = orig_box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_param[mode + '_factor'])

                if self.center_jitter_param.get(mode + '_mode', 'uniform') == 'uniform':
                    max_offset = (jittered_size.prod().sqrt() * self.center_jitter_param[mode + '_factor']).item()
                    offset_factor = (torch.rand(2) - 0.5)
                    jittered_center = orig_box[0:2] + 0.5 * orig_box[2:4] + max_offset * offset_factor

                    if self.center_jitter_param.get(mode + '_limit_motion', False) and i > 0:
                        prev_out_box_center = out_boxes[-1][:2] + 0.5 * out_boxes[-1][2:]
                        if abs(jittered_center[0] - prev_out_box_center[0]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[0] = orig_box[0] + 0.5 * orig_box[2] + max_offset * offset_factor[0] * -1

                        if abs(jittered_center[1] - prev_out_box_center[1]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[1] = orig_box[1] + 0.5 * orig_box[3] + max_offset * offset_factor[1] * -1

                jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

                if self._check_if_crop_inside_image(jittered_box, images[i].shape):
                    break
                else:
                    jittered_box = torch.tensor([1, 1, 10, 10]).float()

            out_boxes.append(jittered_box)

        return out_boxes

    def _generate_proposals(self, frame2_gt_crop):
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        frame2_proposals = np.zeros((num_proposals, 4))
        gt_iou = np.zeros(num_proposals)
        sample_p = np.zeros(num_proposals)

        for i in range(num_proposals):
            frame2_proposals[i, :], gt_iou[i], sample_p[i] = prutils.perturb_box(
                frame2_gt_crop,
                min_iou=self.proposal_params['min_iou'],
                sigma_factor=self.proposal_params['sigma_factor']
            )

        gt_iou = gt_iou * 2 - 1

        return frame2_proposals, gt_iou

    def _generate_label_function(self, target_bb, target_absent=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))
        if target_absent is not None:
            gauss_label *= (1 - target_absent).view(-1, 1, 1).float()
        return gauss_label

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            # Generate synthetic sequence
            jittered_anno = self._generate_synthetic_motion(data[s + '_anno'], data[s + '_images'], s)

            # Crop images
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.output_sz)

            # Add transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a.numpy()) for a in data['test_anno']])

            data['test_proposals'] = [torch.tensor(p, dtype=torch.float32) for p in frame2_proposals]
            data['proposal_iou'] = [torch.tensor(gi, dtype=torch.float32) for gi in gt_iou]

        data = data.apply(stack_tensors)

        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            test_target_absent = 1 - (data['test_visible'] * data['test_valid_anno'])

            data['test_label'] = self._generate_label_function(data['test_anno'], test_target_absent)

        return data

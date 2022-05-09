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


class TargetCandiateMatchingProcessing(BaseProcessing):
    """ The processing class used for training KeepTrack. The distractor dataset for LaSOT is required.
        Two different modes are available partial supervision (partial_sup) or self-supervision (self_sup).

        For partial supervision the candidates their meta data and the images of two consecutive frames are used to
        form a single supervision cue among the candidates corresponding to the annotated target object. All other
        candidates are ignored. First, the search area region is cropped from the image followed by augmentation.
        Then, the candidate matching with the annotated target object is detected to supervise the matching. Then, the
        score map coordinates of the candidates are transformed to full image coordinates. Next, it is randomly decided
        whether the candidates corresponding to the target is dropped in one of the frames to simulate re-detection,
        occlusions or normal tracking. To enable training in batches the number of candidates to match between
        two frames is fixed. Hence, artificial candidates are added. Finally, the assignment matrix is formed where a 1
        denotes a match between two candidates, -1 denotes that a match is not available and -2 denotes that no
        information about the matching is available. These entries will be ignored.

        The second method for partial supervision is used for validation only. It uses only the detected candidates and
        thus results in different numbers of candidates for each frame-pair such that training in batches is not possible.

        For self-supervision only a singe frame and its candidates are required. The second frame and candidates are
        artificially created using augmentations. Here full supervision among all candidates is enabled.
        First, the search area region is cropped from the full image. Then, the cropping coordinates are augmented to
        crop a slightly different view that mimics search area region of the next frame.
        Next, the two image regions are augmented further. Then, the matching between candidates is determined by randomly
        dropping candidates to mimic occlusions or re-detections. Again, the number of candidates is fixed by adding
        artificial candidates that are ignored during training. In addition, the scores  and coordinates of each
        candidate are altered to increase matching difficulty. Finally, the assignment matrix is formed where a 1
        denotes a match between two candidates, -1 denotes that a match is not available.
        """

    def __init__(self, output_sz, num_target_candidates=None, mode='self_sup',
                 img_aug_transform=None, score_map_sz=None, enable_search_area_aug=True,
                 search_area_jitter_value=100, real_target_candidates_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.num_target_candidates = num_target_candidates
        self.mode = mode
        self.img_aug_transform = img_aug_transform
        self.enable_search_area_aug = enable_search_area_aug
        self.search_area_jitter_value = search_area_jitter_value
        self.real_target_candidates_only = real_target_candidates_only
        self.score_map_sz = score_map_sz if score_map_sz is not None else (23, 23)

    def __call__(self, data: TensorDict):
        if data['sup_mode'] == 'self_sup':
            data = self._original_and_augmented_frame(data)
        elif data['sup_mode'] == 'partial_sup' and self.real_target_candidates_only == False:
            data = self._previous_and_current_frame(data)
        elif data['sup_mode'] == 'partial_sup' and self.real_target_candidates_only == True:
            data = self._previous_and_current_frame_detected_target_candidates_only(data)
        else:
            raise NotImplementedError()

        data = data.apply(stack_tensors)

        return data

    def _original_and_augmented_frame(self, data: TensorDict):
        out = TensorDict()
        img = data.pop('img')[0]
        tsm_coords = data['target_candidate_coords'][0]
        scores = data['target_candidate_scores'][0]
        sa_box = data['search_area_box'][0]
        sa_box0 = sa_box.clone()
        sa_box1 = sa_box.clone()

        out['img_shape0'] = [torch.tensor(img.shape[:2])]
        out['img_shape1'] = [torch.tensor(img.shape[:2])]

        # prepared cropped image
        frame_crop0 = prutils.sample_target_from_crop_region(img, sa_box0, self.output_sz)

        x, y, w, h = sa_box.long().tolist()

        if self.enable_search_area_aug:
            l = self.search_area_jitter_value
            sa_box1 = torch.tensor([x + torch.randint(-w//l, w//l+1, (1,)),
                                    y + torch.randint(-h//l, h//l+1, (1,)),
                                    w + torch.randint(-w//l, w//l+1, (1,)),
                                    h + torch.randint(-h//l, h//l+1, (1,))])

        frame_crop1 = prutils.sample_target_from_crop_region(img, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.img_aug_transform(image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        x, y, w, h = sa_box0.tolist()
        img_coords = torch.stack([
            h * (tsm_coords[:, 0].float() / (self.score_map_sz[0] - 1)) + y,
            w * (tsm_coords[:, 1].float() / (self.score_map_sz[1] - 1)) + x
        ]).permute(1, 0)

        img_coords_pad0, img_coords_pad1, valid0, valid1 = self._candidate_drop_out(img_coords, img_coords.clone())

        img_coords_pad0, img_coords_pad1 = self._pad_with_fake_candidates(img_coords_pad0, img_coords_pad1, valid0, valid1,
                                                                          sa_box0, sa_box1, img.shape)

        scores_pad0 = self._add_fake_candidate_scores(scores, valid0)
        scores_pad1 = self._add_fake_candidate_scores(scores, valid1)

        x0, y0, w0, h0 = sa_box0.long().tolist()

        tsm_coords_pad0 = torch.stack([
            torch.round((img_coords_pad0[:, 0] - y0) / h0 * (self.score_map_sz[0] - 1)).long(),
            torch.round((img_coords_pad0[:, 1] - x0) / w0 * (self.score_map_sz[1] - 1)).long()
        ]).permute(1, 0)

        # make sure that the augmented search_are_box is only used for the fake img_coords the other need the original.
        x1, y1, w1, h1 = sa_box1.long().tolist()
        y = torch.where(valid1 == 1, torch.tensor(y0), torch.tensor(y1))
        x = torch.where(valid1 == 1, torch.tensor(x0), torch.tensor(x1))
        h = torch.where(valid1 == 1, torch.tensor(h0), torch.tensor(h1))
        w = torch.where(valid1 == 1, torch.tensor(w0), torch.tensor(w1))

        tsm_coords_pad1 = torch.stack([
            torch.round((img_coords_pad1[:, 0] - y) / h * (self.score_map_sz[0] - 1)).long(),
            torch.round((img_coords_pad1[:, 1] - x) / w * (self.score_map_sz[1] - 1)).long()
        ]).permute(1, 0)

        assert torch.all(tsm_coords_pad0 >= 0) and torch.all(tsm_coords_pad0 < self.score_map_sz[0])
        assert torch.all(tsm_coords_pad1 >= 0) and torch.all(tsm_coords_pad1 < self.score_map_sz[0])

        img_coords_pad1 = self._augment_coords(img_coords_pad1, img.shape, sa_box1)
        scores_pad1 = self._augment_scores(scores_pad1, valid1, ~torch.all(valid0 == valid1))

        out['candidate_img_coords0'] = [img_coords_pad0]
        out['candidate_img_coords1'] = [img_coords_pad1]
        out['candidate_tsm_coords0'] = [tsm_coords_pad0]
        out['candidate_tsm_coords1'] = [tsm_coords_pad1]
        out['candidate_scores0'] = [scores_pad0]
        out['candidate_scores1'] = [scores_pad1]
        out['candidate_valid0'] = [valid0]
        out['candidate_valid1'] = [valid1]

        # Prepare gt labels

        gt_assignment = torch.zeros((self.num_target_candidates, self.num_target_candidates))
        gt_assignment[torch.arange(self.num_target_candidates), torch.arange(self.num_target_candidates)] = valid0 * valid1

        gt_matches0 = torch.arange(0, self.num_target_candidates).float()
        gt_matches1 = torch.arange(0, self.num_target_candidates).float()

        gt_matches0[(valid0==0) | (valid1==0)] = -1
        gt_matches1[(valid0==0) | (valid1==0)] = -1

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out

    def _previous_and_current_frame(self, data: TensorDict):
        out = TensorDict()
        imgs = data.pop('img')
        img0 = imgs[0]
        img1 = imgs[1]
        sa_box0 = data['search_area_box'][0]
        sa_box1 = data['search_area_box'][1]
        tsm_anno_coord0 = data['target_anno_coord'][0]
        tsm_anno_coord1 = data['target_anno_coord'][1]
        tsm_coords0 = data['target_candidate_coords'][0]
        tsm_coords1 = data['target_candidate_coords'][1]
        scores0 = data['target_candidate_scores'][0]
        scores1 = data['target_candidate_scores'][1]

        out['img_shape0'] = [torch.tensor(img0.shape[:2])]
        out['img_shape1'] = [torch.tensor(img1.shape[:2])]

        frame_crop0 = prutils.sample_target_from_crop_region(img0, sa_box0, self.output_sz)
        frame_crop1 = prutils.sample_target_from_crop_region(img1, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.transform['train'](image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        gt_idx0 = self._find_gt_candidate_index(tsm_coords0, tsm_anno_coord0)
        gt_idx1 = self._find_gt_candidate_index(tsm_coords1, tsm_anno_coord1)

        x0, y0, w0, h0 = sa_box0.tolist()
        x1, y1, w1, h1 = sa_box1.tolist()

        img_coords0 = torch.stack([
            h0 * (tsm_coords0[:, 0].float() / (self.score_map_sz[0] - 1)) + y0,
            w0 * (tsm_coords0[:, 1].float() / (self.score_map_sz[1] - 1)) + x0
        ]).permute(1, 0)

        img_coords1 = torch.stack([
            h1 * (tsm_coords1[:, 0].float() / (self.score_map_sz[0] - 1)) + y1,
            w1 * (tsm_coords1[:, 1].float() / (self.score_map_sz[1] - 1)) + x1
        ]).permute(1, 0)

        frame_id, dropout = self._gt_candidate_drop_out()

        drop0 = dropout & (frame_id == 0)
        drop1 = dropout & (frame_id == 1)

        img_coords_pad0, valid0 = self._pad_with_fake_candidates_drop_gt(img_coords0, drop0, gt_idx0, sa_box0, img0.shape)
        img_coords_pad1, valid1 = self._pad_with_fake_candidates_drop_gt(img_coords1, drop1, gt_idx1, sa_box1, img1.shape)

        scores_pad0 = self._add_fake_candidate_scores(scores0, valid0)
        scores_pad1 = self._add_fake_candidate_scores(scores1, valid1)

        x0, y0, w0, h0 = sa_box0.long().tolist()
        x1, y1, w1, h1 = sa_box1.long().tolist()


        tsm_coords_pad0 = torch.stack([
            torch.round((img_coords_pad0[:, 0] - y0) / h0 * (self.score_map_sz[0] - 1)).long(),
            torch.round((img_coords_pad0[:, 1] - x0) / w0 * (self.score_map_sz[1] - 1)).long()
        ]).permute(1, 0)

        tsm_coords_pad1 = torch.stack([
            torch.round((img_coords_pad1[:, 0] - y1) / h1 * (self.score_map_sz[0] - 1)).long(),
            torch.round((img_coords_pad1[:, 1] - x1) / w1 * (self.score_map_sz[1] - 1)).long()
        ]).permute(1, 0)

        assert torch.all(tsm_coords_pad0 >= 0) and torch.all(tsm_coords_pad0 < self.score_map_sz[0])
        assert torch.all(tsm_coords_pad1 >= 0) and torch.all(tsm_coords_pad1 < self.score_map_sz[0])

        out['candidate_img_coords0'] = [img_coords_pad0]
        out['candidate_img_coords1'] = [img_coords_pad1]
        out['candidate_tsm_coords0'] = [tsm_coords_pad0]
        out['candidate_tsm_coords1'] = [tsm_coords_pad1]
        out['candidate_scores0'] = [scores_pad0]
        out['candidate_scores1'] = [scores_pad1]
        out['candidate_valid0'] = [valid0]
        out['candidate_valid1'] = [valid1]

        # Prepare gt labels
        gt_assignment = torch.zeros((self.num_target_candidates, self.num_target_candidates))
        gt_assignment[gt_idx0, gt_idx1] = valid0[gt_idx0]*valid1[gt_idx1]

        gt_matches0 = torch.zeros(self.num_target_candidates) - 2
        gt_matches1 = torch.zeros(self.num_target_candidates) - 2

        if drop0:
            gt_matches0[gt_idx0] = -2
            gt_matches1[gt_idx1] = -1
        elif drop1:
            gt_matches0[gt_idx0] = -1
            gt_matches0[gt_idx1] = -2
        else:
            gt_matches0[gt_idx0] = gt_idx1
            gt_matches1[gt_idx1] = gt_idx0

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out

    def _previous_and_current_frame_detected_target_candidates_only(self, data: TensorDict):
        out = TensorDict()
        imgs = data.pop('img')
        img0 = imgs[0]
        img1 = imgs[1]
        sa_box0 = data['search_area_box'][0]
        sa_box1 = data['search_area_box'][1]
        tsm_anno_coord0 = data['target_anno_coord'][0]
        tsm_anno_coord1 = data['target_anno_coord'][1]
        tsm_coords0 = data['target_candidate_coords'][0]
        tsm_coords1 = data['target_candidate_coords'][1]
        scores0 = data['target_candidate_scores'][0]
        scores1 = data['target_candidate_scores'][1]

        out['img_shape0'] = [torch.tensor(img0.shape[:2])]
        out['img_shape1'] = [torch.tensor(img1.shape[:2])]

        frame_crop0 = prutils.sample_target_from_crop_region(img0, sa_box0, self.output_sz)
        frame_crop1 = prutils.sample_target_from_crop_region(img1, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.transform['train'](image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        gt_idx0 = self._find_gt_candidate_index(tsm_coords0, tsm_anno_coord0)
        gt_idx1 = self._find_gt_candidate_index(tsm_coords1, tsm_anno_coord1)

        x0, y0, w0, h0 = sa_box0.tolist()
        x1, y1, w1, h1 = sa_box1.tolist()

        img_coords0 = torch.stack([
            h0 * (tsm_coords0[:, 0].float() / (self.score_map_sz[0] - 1)) + y0,
            w0 * (tsm_coords0[:, 1].float() / (self.score_map_sz[1] - 1)) + x0
        ]).permute(1, 0)

        img_coords1 = torch.stack([
            h1 * (tsm_coords1[:, 0].float() / (self.score_map_sz[0] - 1)) + y1,
            w1 * (tsm_coords1[:, 1].float() / (self.score_map_sz[1] - 1)) + x1
        ]).permute(1, 0)

        out['candidate_img_coords0'] = [img_coords0]
        out['candidate_img_coords1'] = [img_coords1]
        out['candidate_tsm_coords0'] = [tsm_coords0]
        out['candidate_tsm_coords1'] = [tsm_coords1]
        out['candidate_scores0'] = [scores0]
        out['candidate_scores1'] = [scores1]
        out['candidate_valid0'] = [torch.ones_like(scores0)]
        out['candidate_valid1'] = [torch.ones_like(scores1)]

        # Prepare gt labels
        gt_assignment = torch.zeros((scores0.shape[0], scores1.shape[0]))
        gt_assignment[gt_idx0, gt_idx1] = 1

        gt_matches0 = torch.zeros(scores0.shape[0]) - 2
        gt_matches1 = torch.zeros(scores1.shape[0]) - 2

        gt_matches0[gt_idx0] = gt_idx1
        gt_matches1[gt_idx1] = gt_idx0

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out

    def _find_gt_candidate_index(self, coords, target_anno_coord):
        gt_idx = torch.argmin(torch.sum((coords - target_anno_coord) ** 2, dim=1))
        return gt_idx

    def _gt_candidate_drop_out(self):
        dropout = (torch.rand(1) < 0.25).item()
        frameid = torch.randint(0, 2, (1,)).item()
        return frameid, dropout

    def _pad_with_fake_candidates_drop_gt(self, img_coords, dropout, gt_idx, sa_box, img_shape):
        H, W = img_shape[:2]
        num_peaks = min(img_coords.shape[0], self.num_target_candidates)
        x, y, w, h = sa_box.long().tolist()

        lowx, lowy, highx, highy = max(0, x), max(0, y), min(W, x + w), min(H, y + h)

        img_coords_pad = torch.zeros((self.num_target_candidates, 2))
        valid = torch.zeros(self.num_target_candidates)

        img_coords_pad[:num_peaks] = img_coords[:num_peaks]
        valid[:num_peaks] = 1

        gt_coords = img_coords_pad[gt_idx].clone().unsqueeze(0)

        if dropout:
            valid[gt_idx] = 0
            img_coords_pad[gt_idx] = 0

        filled = valid.clone()
        for i in range(0, self.num_target_candidates):
            if filled[i] == 0:
                cs = torch.cat([
                    torch.rand((20, 1)) * (highy - lowy) + lowy,
                    torch.rand((20, 1)) * (highx - lowx) + lowx
                ], dim=1)

                cs_used = torch.cat([img_coords_pad[filled == 1], gt_coords], dim=0)

                dist = torch.sqrt(torch.sum((cs_used[:, None, :] - cs[None, :, :]) ** 2, dim=2))
                min_dist = torch.min(dist, dim=0).values
                max_min_dist_idx = torch.argmax(min_dist)
                img_coords_pad[i] = cs[max_min_dist_idx]
                filled[i] = 1

        return img_coords_pad, valid

    def _candidate_drop_out(self, coords0, coords1):
        num_candidates = min(coords1.shape[0], self.num_target_candidates)
        num_candidates_to_drop = torch.round(0.25*num_candidates*torch.rand(1)).long()
        idx = torch.randperm(num_candidates)[:num_candidates_to_drop]

        coords_pad0 = torch.zeros((self.num_target_candidates, 2))
        valid0 = torch.zeros(self.num_target_candidates)
        coords_pad1 = torch.zeros((self.num_target_candidates, 2))
        valid1 = torch.zeros(self.num_target_candidates)

        coords_pad0[:num_candidates] = coords0[:num_candidates]
        coords_pad1[:num_candidates] = coords1[:num_candidates]

        valid0[:num_candidates] = 1
        valid1[:num_candidates] = 1

        if torch.rand(1) < 0.5:
            coords_pad0[idx] = 0
            valid0[idx] = 0
        else:
            coords_pad1[idx] = 0
            valid1[idx] = 0

        return coords_pad0, coords_pad1, valid0, valid1

    def _pad_with_fake_candidates(self, img_coords_pad0, img_coords_pad1, valid0, valid1, sa_box0, sa_box1, img_shape):
        H, W = img_shape[:2]

        x0, y0, w0, h0 = sa_box0.long().tolist()
        x1, y1, w1, h1 = sa_box1.long().tolist()

        lowx = [max(0, x0), max(0, x1)]
        lowy = [max(0, y0), max(0, y1)]
        highx = [min(W, x0 + w0), min(W, x1 + w1)]
        highy = [min(H, y0 + h0), min(H, y1 + h1)]

        filled = [valid0.clone(), valid1.clone()]
        img_coords_pad = [img_coords_pad0.clone(), img_coords_pad1.clone()]

        for i in range(0, self.num_target_candidates):
            for k in range(0, 2):
                if filled[k][i] == 0:
                    cs = torch.cat([
                        torch.rand((20, 1)) * (highy[k] - lowy[k]) + lowy[k],
                        torch.rand((20, 1)) * (highx[k] - lowx[k]) + lowx[k]
                    ], dim=1)

                    cs_used = torch.cat([img_coords_pad[0][filled[0]==1], img_coords_pad[1][filled[1]==1]], dim=0)

                    dist = torch.sqrt(torch.sum((cs_used[:, None, :] - cs[None, :, :]) ** 2, dim=2))
                    min_dist = torch.min(dist, dim=0).values
                    max_min_dist_idx = torch.argmax(min_dist)
                    img_coords_pad[k][i] = cs[max_min_dist_idx]
                    filled[k][i] = 1

        return img_coords_pad[0], img_coords_pad[1]

    def _add_fake_candidate_scores(self, scores, valid):
        scores_pad = torch.zeros(valid.shape[0])
        scores_pad[valid == 1] = scores[:self.num_target_candidates][valid[:scores.shape[0]] == 1]
        scores_pad[valid == 0] = (torch.abs(torch.randn((valid==0).sum()))/50).clamp_max(0.025) + 0.05
        return scores_pad

    def _augment_scores(self, scores, valid, drop):
        num_valid = (valid==1).sum()

        noise = 0.1 * torch.randn(num_valid)

        if num_valid > 2 and not drop:
            if scores[1] > 0.5*scores[0] and torch.all(scores[:2] > 0.2):
                # two valid peaks with a high score that are relatively close.
                mode = torch.randint(0, 3, size=(1,))
                if mode == 0:
                    # augment randomly.
                    scores_aug = torch.sort(noise + scores[valid==1], descending=True)[0]
                elif mode == 1:
                    # move peaks closer
                    scores_aug = torch.sort(noise + scores[valid == 1], descending=True)[0]
                    scores_aug[0] = scores[valid==1][0] - torch.abs(noise[0])
                    scores_aug[1] = scores[valid==1][1] + torch.abs(noise[1])
                    scores_aug[:2] = torch.sort(scores_aug[:2], descending=True)[0]
                else:
                    # move peaks closer and switch
                    scores_aug = torch.sort(noise + scores[valid == 1], descending=True)[0]
                    scores_aug[0] = scores[valid==1][0] - torch.abs(noise[0])
                    scores_aug[1] = scores[valid==1][1] + torch.abs(noise[1])
                    scores_aug[:2] = torch.sort(scores_aug[:2], descending=True)[0]

                    idx = torch.arange(num_valid)
                    idx[:2] = torch.tensor([1, 0])
                    scores_aug = scores_aug[idx]
            else:
                scores_aug = torch.sort(scores[valid==1] + noise, descending=True)[0]

        else:
            scores_aug = torch.sort(scores[valid == 1] + noise, descending=True)[0]

        scores_aug = scores_aug.clamp_min(0.075)

        scores[valid==1] = scores_aug.clone()

        return scores

    def _augment_coords(self, coords, img_shape, search_area_box):
        H, W = img_shape[:2]

        _, _, w, h = search_area_box.float()

        # add independent offset to each coord
        d = torch.sqrt(torch.sum((coords[None, :] - coords[:, None])**2, dim=2))

        if torch.all(d == 0):
            xmin = 0.5*w/self.score_map_sz[1]
            ymin = 0.5*h/self.score_map_sz[0]
        else:
            dmin = torch.min(d[d>0])
            xmin = (math.sqrt(2)*dmin/4).clamp_max(w/self.score_map_sz[1])
            ymin = (math.sqrt(2)*dmin/4).clamp_max(h/self.score_map_sz[0])

        txi = torch.rand(coords.shape[0])*2*xmin - xmin
        tyi = torch.rand(coords.shape[0])*2*ymin - ymin

        coords[:, 0] += tyi
        coords[:, 1] += txi

        coords[:, 0] = coords[:, 0].clamp(0, H)
        coords[:, 1] = coords[:, 1].clamp(0, W)

        return coords


class LTRBDenseRegressionProcessing(BaseProcessing):
    """ The processing class used for training ToMP that supports dense bounding box regression.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', stride=16, label_function_params=None,
                 center_sampling_radius=0.0, use_normalized_coords=True, *args, **kwargs):
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
        self.stride = stride
        self.label_function_params = label_function_params
        self.center_sampling_radius = center_sampling_radius
        self.use_normalized_coords = use_normalized_coords

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

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label

    def _generate_ltbr_regression_targets(self, target_bb):
        shifts_x = torch.arange(
            0, self.output_sz, step=self.stride,
            dtype=torch.float32, device=target_bb.device
        )
        shifts_y = torch.arange(
            0, self.output_sz, step=self.stride,
            dtype=torch.float32, device=target_bb.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + self.stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([target_bb[:, 0], target_bb[:, 1], target_bb[:, 0] + target_bb[:, 2],
                            target_bb[:, 1] + target_bb[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        if self.use_normalized_coords:
            reg_targets_per_im = reg_targets_per_im / self.output_sz

        if self.center_sampling_radius > 0:
            is_in_box = self._compute_sampling_region(xs, xyxy, ys)
        else:
            is_in_box = (reg_targets_per_im.min(dim=1)[0] > 0)

        sz = self.output_sz//self.stride
        nb = target_bb.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)
        is_in_box = is_in_box.reshape(sz, sz, nb, 1).permute(2, 3, 0, 1)

        return reg_targets_per_im, is_in_box

    def _compute_sampling_region(self, xs, xyxy, ys):
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xmin = cx - self.center_sampling_radius * self.stride
        ymin = cy - self.center_sampling_radius * self.stride
        xmax = cx + self.center_sampling_radius * self.stride
        ymax = cy + self.center_sampling_radius * self.stride
        center_gt = xyxy.new_zeros(xyxy.shape)
        center_gt[:, 0] = torch.where(xmin > xyxy[:, 0], xmin, xyxy[:, 0])
        center_gt[:, 1] = torch.where(ymin > xyxy[:, 1], ymin, xyxy[:, 1])
        center_gt[:, 2] = torch.where(xmax > xyxy[:, 2], xyxy[:, 2], xmax)
        center_gt[:, 3] = torch.where(ymax > xyxy[:, 3], xyxy[:, 3], ymax)
        left = xs[:, None] - center_gt[:, 0]
        right = center_gt[:, 2] - xs[:, None]
        top = ys[:, None] - center_gt[:, 1]
        bottom = center_gt[:, 3] - ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_box = center_bbox.min(-1)[0] > 0
        return is_in_box

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
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'],
                                                                             bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

        data['test_ltrb_target'], data['test_sample_region'] = self._generate_ltbr_regression_targets(data['test_anno'])
        data['train_ltrb_target'], data['train_sample_region'] = self._generate_ltbr_regression_targets(data['train_anno'])

        return data

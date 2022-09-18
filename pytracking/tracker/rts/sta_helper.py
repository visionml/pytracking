import torch
import torch.nn.functional as F
import numpy as np
import math
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch


class STAHelper:

    def __init__(self, params):
        self.params = params

        # The whole network
        self.net = self.params.sta_net

        # Set sizes
        sz = self.params.sta_image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area.
        self.search_area_scale = self.params.sta_search_area_scale
        # Extract and transform sample
        self.feature_sz = self.img_sample_sz / 16
        ksz = self.net.target_model.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

    def predict_mask(self, image, bbox):

        invalid_bbox = (bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1)
        if invalid_bbox:
            print("Initial bounding box is invalid. This should not happen!")
            exit(1)

        # Convert image
        all_images = [image, ]
        all_images = [numpy_to_torch(a_im) for a_im in all_images]

        test_bbox = torch.from_numpy(bbox)
        all_boxes = test_bbox.view(1, -1)
        im_patches, sample_coords, box_patches = self.extract_image_crops(all_images, all_boxes, self.img_sample_sz)

        # predict segmentation masks
        im_patches = torch.cat(im_patches, dim=0).to(self.params.device).unsqueeze(1)
        box_patches = box_patches.unsqueeze(1).to(self.params.device)

        with torch.no_grad():
            _, segmentation_scores = self.net.forward(im_patches, box_patches)

        # Location of sample
        sample_pos_test, sample_scale_test = self.get_sample_location(sample_coords[0][0])
        # Get the segmentation scores for the full image.
        # Regions outside the search region are assigned low scores (-100)
        segmentation_scores_im_test = self.convert_scores_crop_to_image(
            segmentation_scores[:1], all_images[0], sample_scale_test, sample_pos_test)

        # Set scores outside the box to be low
        test_bbox_np = test_bbox.numpy().astype(np.int64)
        segmentation_scores_im_test[..., :test_bbox_np[0]] = -100
        segmentation_scores_im_test[..., test_bbox_np[0]+test_bbox_np[2]:] = -100
        segmentation_scores_im_test[..., :test_bbox_np[1], :] = -100
        segmentation_scores_im_test[..., test_bbox_np[1]+test_bbox_np[3]:, :] = -100

        segmentation_mask_im_test = (segmentation_scores_im_test > 0.0).float()
        segmentation_mask_im_test = segmentation_mask_im_test.view(*segmentation_mask_im_test.shape[-2:])

        return segmentation_mask_im_test


    def get_box_in_crop_coords(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame"""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:2] + sample_coord[2:] - 1)
        sample_scales = ((sample_coord[2:] - sample_coord[:2]) / self.img_sample_sz).prod().sqrt()
        return sample_pos, sample_scales

    def convert_scores_crop_to_image(self, segmentation_scores, im, sample_scale, sample_pos):
        """ Obtain segmentation scores for the full image using the scores for the search region crop. This is done by
            assigning a low score (-100) for image regions outside the search region """

        # Resize the segmention scores to match the image scale
        segmentation_scores_re = F.interpolate(segmentation_scores, scale_factor=sample_scale.item(), mode='bilinear')
        segmentation_scores_re = segmentation_scores_re.view(*segmentation_scores_re.shape[-2:])

        # Regions outside search area get very low score
        segmentation_scores_im = torch.ones(im.shape[-2:], dtype=segmentation_scores_re.dtype) * (-100.0)

        # Find the co-ordinates of the search region in the image scale
        r1 = int(sample_pos[0].item() - 0.5*segmentation_scores_re.shape[-2])
        c1 = int(sample_pos[1].item() - 0.5*segmentation_scores_re.shape[-1])

        r2 = r1 + segmentation_scores_re.shape[-2]
        c2 = c1 + segmentation_scores_re.shape[-1]

        r1_pad = max(0, -r1)
        c1_pad = max(0, -c1)

        r2_pad = max(r2 - im.shape[-2], 0)
        c2_pad = max(c2 - im.shape[-1], 0)

        # Copy the scores for the search region
        shape = segmentation_scores_re.shape
        segmentation_scores_im[r1 + r1_pad:r2 - r2_pad, c1 + c1_pad:c2 - c2_pad] = \
            segmentation_scores_re[r1_pad:shape[0] - r2_pad, c1_pad:shape[1] - c2_pad]

        return segmentation_scores_im

    def extract_image_crops(self, images, boxes, image_sz):
        im_patches = []
        patch_coords = []
        box_patches = []

        for im, box in zip(images, boxes):
            pos = torch.Tensor([box[1] + (box[3] - 1) / 2, box[0] + (box[2] - 1) / 2])

            target_sz = torch.Tensor([box[3], box[2]])

            search_area = torch.prod(target_sz * self.search_area_scale).item()
            target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

            im_patch, patch_coord = sample_patch(im, pos, target_scale * image_sz, image_sz,
                                                 mode=self.params.get('sta_border_mode', 'replicate'),
                                                 max_scale_change=self.params.get('sta_patch_max_scale_change', None))
            sample_pos, sample_scale = self.get_sample_location(patch_coord[0])
            box_patch = self.get_box_in_crop_coords(pos, target_sz, sample_pos, sample_scale)

            im_patches.append(im_patch)
            patch_coords.append(patch_coord)
            box_patches.append(box_patch)

        box_patches = torch.stack(box_patches, dim=0)
        return im_patches, patch_coords, box_patches

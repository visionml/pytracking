import os
import numpy as np
import copy
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader, imread_indexed
import torch
import random
from collections import OrderedDict
from ltr.admin.environment import env_settings
import pandas as pd
from PIL import Image
import torch.nn.functional as F


class OpenImages(BaseImageDataset):
    """ The Open Images dataset. Open Images is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:

    Download the images along with annotations from https://storage.googleapis.com/openimages/web/download.html. The root
    folder should be organized as follows.
        - open_images_root
            - annotations
                - train-annotations-bbox.csv
                - train-annotations-object-segmentation.csv
                - validation-annotations-bbox.csv
                - validation-annotations-object-segmentation.csv
            - images
                - train_{0-9, a-f}
                - validation
            - masks
                - train_{0-9, a-f}
                - validation

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", set_ids=None):
        """
        args:
            root        - The path to the OpenImages folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the OpenImages sets to be used for training. If None, all the
                            sets (0 - 9, a - f) will be used.
        """
        root = env_settings().open_images_dir if root is None else root
        super().__init__('OpenImages', root, image_loader)

        assert split in ["train", "validation", "test"]
        self.split = split

        if set_ids is None:
            set_ids = list('0123456789abcdef')

        self.set_ids = set_ids

        self.img_pth = os.path.join(root, 'images')
        self.mask_pth = os.path.join(root, 'masks')
        bbox_anno_path = os.path.join(root, 'annotations', f'{split}-annotations-bbox.csv')
        mask_anno_path = os.path.join(root, 'annotations', f'{split}-annotations-object-segmentation.csv')

        # bbox_annotations = pd.read_csv(bbox_anno_path)
        # img_ids = False
        # for set_id in set_ids:
        #     img_ids |= bbox_annotations.ImageID.str.startswith(set_id)
        # self.bbox_annotations = bbox_annotations[img_ids]

        mask_annotations = pd.read_csv(mask_anno_path)
        img_ids = False
        for set_id in set_ids:
            img_ids |= mask_annotations.ImageID.str.startswith(set_id)
        self.mask_annotations = mask_annotations[img_ids]

        self.class_list = self.get_class_list()     # the parent class thing would happen in the sampler

        self.image_list = self._get_image_list()

        if data_fraction is not None:
            self.image_list = random.sample(self.image_list, int(len(self.image_list)*data_fraction))

        print("%s loaded." % self.get_name())

    def _get_image_list(self):
        im_list = self.mask_annotations.index.to_list()
        # TODO: perhaps filter masks that are truncated / too large
        return im_list

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'open_images'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_names = sorted(list(self.mask_annotations['LabelName'].unique()))
        return class_names

    def has_segmentation_info(self):
        return True

    def get_image_info(self, im_id):
        anno = self.mask_annotations.loc[self.image_listt[im_id]]
        image_id = anno['ImageID']

        img_dir = self.split
        if self.split == "train":
            set_id = image_id[0]
            img_dir += f"_{set_id}"
        mask_path = os.path.join(self.mask_pth, img_dir, anno['MaskPath'])

        mask = imread_indexed(mask_path)
        mask = torch.Tensor(mask).unsqueeze(dim=0) / 255
        # Size mismatch between image and mask: https://github.com/openimages/dataset/issues/92
        # Resize the mask to the original image
        w, h = self._image_size(image_id)
        mask = F.interpolate(mask[None], (h, w), mode='bilinear', align_corners=False).squeeze(dim=0)

        bbox = torch.Tensor(self._convert_bbox_format(anno, mask.shape[-2:]))

        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _convert_bbox_format(self, anno, size):
        h, w = size
        bbox = [anno['BoxXMin'] * w, anno['BoxYMin'] * h, anno['BoxXMax'] * w, anno['BoxYMax'] * h]
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    def _image_size(self, image_id):
        img_dir = self.split
        if self.split == "train":
            set_id = image_id[0]
            img_dir += f"_{set_id}"
        path = os.path.join(self.img_pth, img_dir, f"{image_id}.jpg")

        with Image.open(path) as img:
            return img.size  # format: w, h

    def _get_image(self, im_id):
        anno = self.mask_annotations.loc[self.image_list[im_id]]
        image_id = anno['ImageID']

        img_dir = self.split
        if self.split == "train":
            set_id = image_id[0]
            img_dir += f"_{set_id}"
        path = os.path.join(self.img_pth, img_dir, f"{image_id}.jpg")
        img = self.image_loader(path)
        return img

    def get_meta_info(self, im_id):
        try:
            anno = self.mask_annotations.loc[self.image_list[im_id]]
            object_meta = OrderedDict({'object_class_name': anno['LabelName'],
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_class_name(self, im_id):
        anno = self.mask_annotations.loc[self.image_list[im_id]]
        return anno['LabelName']

    def get_image(self, image_id, anno=None):
        image = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return image, anno, object_meta

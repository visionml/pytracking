import os
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader, imread_indexed
import torch
from collections import OrderedDict
from ltr.admin.environment import env_settings
from ltr.data.bounding_box_utils import masks_to_bboxes


class MSRA10k(BaseImageDataset):
    """
    MSRA10k salient object detection dataset

    Publication:
        Global contrast based salient region detection
        Ming-Ming Cheng, Niloy J. Mitra, Xiaolei Huang, Philip H. S. Torr, and Shi-Min Hu
        TPAMI, 2015
        https://mmcheng.net/mftp/Papers/SaliencyTPAMI.pdf

    Download dataset from https://mmcheng.net/msra10k/
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, min_area=None):
        """
        args:
            root - path to MSRA10k root folder
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            min_area - Objects with area less than min_area are filtered out. Default is 0.0
        """
        root = env_settings().msra10k_dir if root is None else root
        super().__init__('MSRA10k', root, image_loader)

        self.image_list = self._load_dataset(min_area=min_area)

        if data_fraction is not None:
            raise NotImplementedError

    def _load_dataset(self, min_area=None):
        files_list = os.listdir(os.path.join(self.root, 'Imgs'))
        image_list = [f[:-4] for f in files_list if f[-3:] == 'jpg']

        images = []

        for f in image_list:
            a = imread_indexed(os.path.join(self.root, 'Imgs', '{}.png'.format(f)))

            if min_area is None or (a > 0).sum() > min_area:
                images.append(f)

        return images

    def get_name(self):
        return 'msra10k'

    def has_segmentation_info(self):
        return True

    def get_image_info(self, im_id):
        mask = imread_indexed(os.path.join(self.root, 'Imgs', '{}.png'.format(self.image_list[im_id])))
        mask = torch.Tensor(mask == 255)
        bbox = masks_to_bboxes(mask, fmt='t').view(4,)

        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def get_meta_info(self, im_id):
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return object_meta

    def get_image(self, image_id, anno=None):
        frame = self.image_loader(os.path.join(self.root, 'Imgs', '{}.jpg'.format(self.image_list[image_id])))

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return frame, anno, object_meta

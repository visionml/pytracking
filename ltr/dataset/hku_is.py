import os
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader, opencv_loader, imread_indexed
import torch
from pycocotools.coco import COCO
import random
from collections import OrderedDict
from ltr.admin.environment import env_settings
from ltr.data.bounding_box_utils import masks_to_bboxes


class HKUIS(BaseImageDataset):
    """

    Args:
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, min_area=None):
        root = env_settings().hkuis_dir if root is None else root
        super().__init__('HKUIS', root, image_loader)

        self.image_list, self.anno_list = self._load_dataset(min_area=min_area)

        if data_fraction is not None:
            raise NotImplementedError

    def _load_dataset(self, min_area=None):
        files_list = os.listdir(os.path.join(self.root, 'imgs'))
        image_list = [f[:-4] for f in files_list]

        images = []
        annos = []

        for f in image_list:
            a = imread_indexed(os.path.join(self.root, 'gt', '{}.png'.format(f)))

            if min_area is None or (a > 0).sum() > min_area:
                im = opencv_loader(os.path.join(self.root, 'imgs', '{}.png'.format(f)))
                images.append(im)
                annos.append(a)

        return images, annos

    def get_name(self):
        return 'hku-is'

    def has_segmentation_info(self):
        return True

    def get_image_info(self, im_id):
        mask = self.anno_list[im_id]
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
        frame = self.image_list[image_id]

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return frame, anno, object_meta

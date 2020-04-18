from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader_w_failsafe
import torch
from collections import OrderedDict
import os
from scipy.io import loadmat
from ltr.data.bounding_box_utils import masks_to_bboxes

from ltr.admin.environment import env_settings


class SBD(BaseImageDataset):
    """
    Semantic Boundaries Dataset and Benchmark (SBD)

    Publication:
        Semantic contours from inverse detectors
        Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji and Jitendra Malik
        ICCV, 2011
        http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf

    Download dataset from: http://home.bharathh.info/pubs/codes/SBD/download.html
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, data_fraction=None, split="train"):
        """
        args:
            root - path to SBD root folder
            image_loader - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                           is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            split - dataset split ("train", "train_noval", "val")
        """
        root = env_settings().sbd_dir if root is None else root
        super().__init__('SBD', root, image_loader)

        assert split in ["train", "train_noval", "val"]

        self.root = root

        self.image_path_list, self.anno_file_list = self._load_dataset(split)

        # Load mat fine
        anno_list = [loadmat(a) for a in self.anno_file_list]

        self.image_list = self._construct_image_list(anno_list)
        if data_fraction is not None:
            raise NotImplementedError

    def _load_dataset(self, split):
        split_f = os.path.join(self.root, split.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        image_list = [os.path.join(self.root, 'img', x + ".jpg") for x in file_names]
        anno_list = [os.path.join(self.root, 'inst', x + ".mat") for x in file_names]

        assert (len(image_list) == len(anno_list))

        return image_list, anno_list

    def _get_mask_from_mat(self, mat):
        return torch.tensor(mat['GTinst'][0]['Segmentation'][0])

    def _construct_image_list(self, anno_list):
        image_list = []

        for im_id, a in enumerate(anno_list):
            mask = self._get_mask_from_mat(a)
            for instance_id in range(1, mask.max().item() + 1):
                image_list.append((im_id, instance_id))

        return image_list

    def get_name(self):
        return 'sbd'

    def has_segmentation_info(self):
        return True

    def get_image_info(self, im_id):
        image_id, instance_id = self.image_list[im_id]
        anno_mat = loadmat(self.anno_file_list[image_id])
        mask = self._get_mask_from_mat(anno_mat)

        mask = (mask == instance_id).float()
        bbox = masks_to_bboxes(mask, fmt='t')
        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _get_image(self, im_id):
        image_id, _ = self.image_list[im_id]

        img = self.image_loader(self.image_path_list[image_id])
        return img

    def get_meta_info(self, im_id):
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

    def get_image(self, image_id, anno=None):
        image = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return image, anno, object_meta

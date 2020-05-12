import os
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader_w_failsafe
import torch
import random
import lvis.lvis as lvis_pk
from collections import OrderedDict
from ltr.admin.environment import env_settings


class LVIS(BaseImageDataset):
    """ The LVIS object detection dataset

    Publication:
        LVIS: A Dataset for Large Vocabulary Instance Segmentation
        Agrim Gupta, Piotr Dollár, and Ross Girshick
        CVPR, 2019
        https://arxiv.org/pdf/1908.03195.pdf

    Download the images along with annotations from https://www.lvisdataset.org/dataset. The root folder should be
    organized as follows.
        - lvis_root
            - annotations
                - lvis_v0.5_train.json
                - lvis_v0.5_val.json
            - images
                - val2017
                - train2017

    Note: You also have to install the lvis Python API from https://github.com/lvis-dataset/lvis-api
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, data_fraction=None, min_area=None, split="train"):
        """
        args:
            root - path to lvis root folder
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            min_area - Objects with area less than min_area are filtered out. Default is 0.0
            split - 'train' or 'val'.
        """
        root = env_settings().lvis_dir if root is None else root
        super().__init__('LVIS', root, image_loader)

        self.img_pth = os.path.join(root, 'images', f'{split}2017/')
        self.anno_path = os.path.join(root, 'annotations', f'lvis_v0.5_{split}.json')

        # Load the LVIS set.
        self.lvis_set = lvis_pk.LVIS(self.anno_path)

        self.cats = self.lvis_set.cats

        self.class_list = self.get_class_list()     # the parent class thing would happen in the sampler

        self.image_list = self._get_image_list(min_area=min_area)

        if data_fraction is not None:
            self.image_list = random.sample(self.image_list, int(len(self.image_list) * data_fraction))
        self.im_per_class = self._build_im_per_class()

    def _get_image_list(self, min_area=None):
        im_list = list(self.lvis_set.anns.keys())  # No 'iscrowd' information in LVIS

        if min_area is not None:
            im_list = [s for s in im_list if self.lvis_set.anns[s]['area'] > min_area]

        return im_list

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'lvis'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def has_segmentation_info(self):
        return True

    def _build_im_per_class(self):
        im_per_class = {}
        for i, im in enumerate(self.image_list):
            class_name = self.cats[self.lvis_set.anns[im]['category_id']]['name']
            if class_name not in im_per_class:
                im_per_class[class_name] = [i]
            else:
                im_per_class[class_name].append(i)

        return im_per_class

    def get_images_in_class(self, class_name):
        return self.im_per_class[class_name]

    def get_image_info(self, im_id):
        anno = self._get_anno(im_id)

        bbox = torch.Tensor(anno['bbox']).view(4,)

        mask = torch.Tensor(self.lvis_set.ann_to_mask(anno))

        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _get_anno(self, im_id):
        anno = self.lvis_set.anns[self.image_list[im_id]]

        return anno

    def _get_image(self, im_id):
        path = self.lvis_set.load_imgs([self.lvis_set.anns[self.image_list[im_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def get_meta_info(self, im_id):
        try:
            cat_dict_current = self.cats[self.lvis_set.anns[self.image_list[im_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': None,  # No 'supercategory' information available in LVIS
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
        cat_dict_current = self.cats[self.lvis_set.anns[self.image_list[im_id]]['category_id']]
        return cat_dict_current['name']

    def get_image(self, image_id, anno=None):
        frame = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return frame, anno, object_meta

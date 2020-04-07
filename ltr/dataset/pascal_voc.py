import os
import numpy as np
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader, imread_indexed
import torch
import random
from collections import OrderedDict
import os
import sys
import collections
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from ltr.admin.environment import env_settings


class PascalVOC(BaseImageDataset):
    """ The Pascal VOC dataset. Pascal VOC is an image dataset. Thus, we treat each image as a sequence of length 1.

    Download the images along with annotations. The root folder should be
    organized as follows.
        - pascal_root
            - VOC2012
                - Annotations
                - JPEGImages
                - ImageSets
                    - Segmentation
                - SegmentationObject
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2012"):
        root = env_settings().pascal_voc_dir if root is None else root
        super().__init__('PascalVOC', root, image_loader)

        assert split in ["train", "trainval", "val"]
        assert version in ['2007', '2012']

        base_dir = f"VOC{version}"
        self.img_pth = os.path.join(root, base_dir, 'JPEGImages')
        self.anno_path = os.path.join(root, base_dir, 'Annotations')
        self.mask_pth = os.path.join(root, base_dir, 'SegmentationObject')

        splits_dir = os.path.join(root, base_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, f'{split}.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.img_pth, f"{x}.jpg") for x in file_names]
        annotation_paths = [os.path.join(self.anno_path, f"{x}.xml") for x in file_names]
        self.masks = [os.path.join(self.mask_pth, f"{x}.png") for x in file_names]

        assert (len(self.images) == len(annotation_paths) == len(self.masks))

        self.anns = []
        for img_id, ann_pth in enumerate(annotation_paths):
            img_anns = self._parse_voc_xml(ET.parse(ann_pth).getroot())
            obj_anns = img_anns['annotation']['object']
            if not isinstance(obj_anns, list):
                obj_anns = [obj_anns]

            for obj_id, ann in enumerate(obj_anns, 1):
                self.anns.append({
                    'image_id': img_id,
                    'object_id': obj_id,
                    'bbox': self._convert_bbox_format(ann['bndbox']),
                    'category': ann['name']
                })

        self.class_list = self.get_class_list()     # the parent class thing would happen in the sampler

        self.image_list = self._get_image_list()

        if data_fraction is not None:
            self.image_list = random.sample(self.image_list, int(len(self.image_list) * data_fraction))

    def _convert_bbox_format(self, b):
        bbox = [int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])]
        return [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]

    def _get_image_list(self):
        return list(range(len(self.anns)))

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'pascal_voc'

    def has_class_info(self):
        return True

    def get_class_list(self):
        return ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def has_segmentation_info(self):
        return True

    def get_image_info(self, im_id):
        anno = self.anns[self.image_list[im_id]]
        bbox = torch.Tensor(anno['bbox'])

        mask = imread_indexed(self.masks[anno['image_id']])
        obj_id = anno['object_id']
        mask = torch.Tensor(mask == obj_id)

        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self._parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def _get_image(self, im_id):
        anno = self.anns[self.image_list[im_id]]
        img = self.image_loader(self.images[anno['image_id']])
        return img

    def get_meta_info(self, im_id):
        try:
            object_meta = OrderedDict({'object_class_name': self.get_class_name(im_id),
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
        anno = self.anns[self.image_list[im_id]]
        return anno['category']

    def get_image(self, image_id, anno=None):
        image = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return image, anno, object_meta

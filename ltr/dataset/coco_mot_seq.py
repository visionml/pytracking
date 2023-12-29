import os
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
import torch
import random
from pycocotools.coco import COCO
from collections import OrderedDict
from ltr.admin.environment import env_settings


class MSCOCOMOTSeq(BaseVideoDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014"):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = env_settings().coco_dir if root is None else root
        super().__init__('COCOMOT', root, image_loader)

        self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))
        self.anno_path = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, version))

        # Load the COCO set.
        self.coco_set = COCO(self.anno_path)

        self.cats = self.coco_set.cats

        self.class_list = self.get_class_list()

        self.sequence_list = self._get_sequence_list()
        self.sequence_list = self._remove_empty_sequences()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def _get_sequence_list(self):
        seq_list = list(self.coco_set.imgs.keys())
        return seq_list

    def _remove_empty_sequences(self):
        num_objects_per_seq = [len(self._get_anno(i)) for i in range(len(self.sequence_list))]
        return [seq for num, seq in zip(num_objects_per_seq, self.sequence_list) if num > 0]

    def is_mot_dataset(self):
        return True

    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'coco_mot'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        bbox = {i: torch.Tensor(anno[i]['bbox']) for i in range(len(anno))}

        return {'bbox': bbox, 'num_tracks': len(anno)}

    def _get_anno(self, seq_id):
        anno = self.coco_set.imgToAnns[self.sequence_list[seq_id]]
        anno_filt = [a for a in anno if a['iscrowd'] == 0]
        return anno_filt

    def _get_frames(self, frame_path):
        img = self.image_loader(frame_path)
        return img

    def _get_frame_path(self, seq_id):
        frame_path = os.path.join(self.img_pth, self.coco_set.imgs[self.sequence_list[seq_id]]['file_name'])
        return frame_path

    def get_meta_info(self, seq_id):
        object_meta = OrderedDict({'object_class_name': None,
                                    'motion_class': None,
                                    'major_class': None,
                                    'root_class': None,
                                    'motion_adverb': None})

        return object_meta


    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None, frames_dict=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame_path = self._get_frame_path(seq_id)
        frame = self._get_frames(frame_path)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if isinstance(value, dict):
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = value

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

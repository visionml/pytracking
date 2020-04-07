import torch
import os
import os.path
import numpy as np
import pandas
import random
import pickle as pkl
from collections import OrderedDict

from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings


class YTBB(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', data_fraction=None):
        base_dir = env_settings().ytbb_dir if root is None else root
        if split == 'train':
            root = os.path.join(base_dir, 'yt_bb_detection_train')
        elif split == 'val':
            root = os.path.join(base_dir, 'yt_bb_detection_validation')
        else:
            raise Exception('Unknown split "{}" for ytbb'.format_map(split))
        
        super().__init__(root, image_loader)

        # Load meta file
        self.sequence_list = self._load_meta_file(root)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        # TODO class info

    def _load_meta_file(self, base_path):
        with open(os.path.join(base_path, 'sequence_data.pkl'), 'rb') as handle:
            data = pkl.load(handle)

        return data

    def get_name(self):
        return 'ytbb'

    def has_class_info(self):
        return True

    def _read_bb_anno(self, seq_id):
        gt = self.sequence_list[seq_id]['bb']

        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.tensor(self.sequence_list[seq_id]['visible']).byte()
        visible = visible & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_id, frame_id):
        class_id = self.sequence_list[seq_id]['class_id']
        vid_name = self.sequence_list[seq_id]['name']
        frame_name = self.sequence_list[seq_id]['frames'][frame_id]
        frame_path = os.path.join(self.root, str(class_id), vid_name, frame_name)
        return self.image_loader(frame_path)

    def _get_class(self, seq_id):
        class_id = self.sequence_list[seq_id]['class_id']
        return class_id

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_class = None #self._get_class(seq_id)

        object_meta = OrderedDict({'object_class': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

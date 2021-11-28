import os
import os.path
import json
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict, defaultdict
from ltr.dataset.base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class LasotCandidateMatching(BaseVideoDataset):
    """ LaSOT dataset dumped results during tracking super_dimp_hinge.
    """

    def __init__(self, root=None, path_to_json=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot candidate matching dataset json file.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dir if root is None else root
        path_to_json = env_settings().lasot_candidate_matching_dataset_path if path_to_json is None else path_to_json
        super().__init__('LaSOTCandidateMatching', root, image_loader)

        self.sequence_info_cache = {}

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

        self.dataset = self._load_dataset(path_to_json)

    def _load_dataset(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def get_frame_states(self):
        frame_states_all = defaultdict(list)

        for seq_id, seq_name in enumerate(self.sequence_list):
            if 'frame_states' in self.dataset[seq_name]:
                for frame_state_name, frame_state_indices in self.dataset[seq_name]['frame_states'].items():
                    frame_state_indices = torch.tensor(frame_state_indices).view(-1, 1)
                    seq_ids = seq_id * torch.ones_like(frame_state_indices).view(-1, 1)
                    data = torch.cat([seq_ids, frame_state_indices], dim=1)
                    frame_states_all[frame_state_name].append(data)

        for state_name in frame_states_all.keys():
            frame_states_all[state_name] = torch.cat(frame_states_all[state_name], dim=0)

        return frame_states_all

    def get_subseq_states(self):
        subseq_states_all = defaultdict(list)

        for seq_id, seq_name in enumerate(self.sequence_list):
            if 'subseq_states' in self.dataset[seq_name]:
                for subseq_state_name, subseq_state_indices in self.dataset[seq_name]['subseq_states'].items():
                    subseq_state_indices = torch.tensor(subseq_state_indices).view(-1, 1)
                    seq_ids = seq_id * torch.ones_like(subseq_state_indices).view(-1, 1)
                    data = torch.cat([seq_ids, subseq_state_indices], dim=1)
                    subseq_states_all[subseq_state_name].append(data)

        for state_name in subseq_states_all.keys():
            subseq_states_all[state_name] = torch.cat(subseq_states_all[state_name], dim=0)

        return subseq_states_all

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_val_split.txt')
            elif split == 'train-train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_train_split.txt')
            elif split == 'train-val':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot_candidate_matching_new'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, 'full_occlusion.txt')
        out_of_view_file = os.path.join(seq_path, 'out_of_view.txt')

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, root, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        if seq_id not in self.sequence_info_cache:
            seq_path_img = self._get_sequence_path(env_settings().lasot_dir, seq_id)
            bbox = self._read_bb_anno(seq_path_img)

            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            visible = self._read_target_visible(seq_path_img) & valid.byte()

            self.sequence_info_cache[seq_id] = dict(bbox=bbox, valid=valid, visible=visible)

        return self.sequence_info_cache[seq_id]

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def _get_data(self, seq_id, seq_img_path, frame_id):
        data = self.dataset[self.get_sequence_name(seq_id)]

        idx = data['index'].index(frame_id)

        search_area_box = torch.FloatTensor(data['search_area_box'][idx])
        target_candidate_scores = torch.FloatTensor(data['target_candidate_scores'][idx])
        target_candidate_coords = torch.FloatTensor(data['target_candidate_coords'][idx])
        target_anno_coord = torch.FloatTensor(data['target_anno_coord'][idx])

        img = self.image_loader(self._get_frame_path(seq_img_path, frame_id))

        return dict(search_area_box=search_area_box, img=img, target_anno_coord=target_anno_coord,
                    target_candidate_coords=target_candidate_coords, target_candidate_scores=target_candidate_scores)

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(self.root, seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_sequence_name(self, seq_id):
        return self.sequence_list[seq_id]

    def get_frames(self, seq_id, frame_ids, anno=None):
        frames_dict = dict()

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        seq_path_img = self._get_sequence_path(env_settings().lasot_dir, seq_id)

        obj_class = self._get_class(seq_path_img)

        dumped_data_frame_list = [self._get_data(seq_id, seq_path_img, f_id) for f_id in frame_ids]

        for key in dumped_data_frame_list[0].keys():
            frames_dict[key] = [data[key] for data in dumped_data_frame_list] # is cloning needed here?

        for key, value in anno.items():
            frames_dict[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frames_dict, object_meta

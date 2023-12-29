import os
import pandas
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
import json
import torch
from collections import OrderedDict
from ltr.admin.environment import env_settings


def get_target_to_image_ratio(seq):
    anno = torch.Tensor(seq['anno'])
    img_sz = torch.Tensor(seq['image_size'])
    return (anno[0, 2:4].prod() / (img_sz.prod())).sqrt()


class ImagenetVIDMOT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, multiobj=True):
        root = env_settings().imagenet_vid_gmot_dir if root is None else root
        super().__init__('ImagenetVIDMOT', root, image_loader)

        path = os.path.join(root, 'ImageNet-Vid.json')
        self.multiobj = multiobj

        with open(path, 'r') as f:
            data = json.load(f)
            self.sequence_list, self.num_tracks_per_seq = data['seq_list'], data['num_tracks']

        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'imagenetvid_mot_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'imagenetvid_mot_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_names = pandas.read_csv(file_path, header=None).squeeze('columns').values.tolist()
            ids = [self.sequence_list.index(s) for s in seq_names]

            self.sequence_list = [(None, self.sequence_list[i]) for i in ids]

        if multiobj == False:
            sequence_list = []
            for i in range(len(self.sequence_list)):
                info = self._preload_sequence_info(i)

                for id in info['bbox'][0].keys():
                    sequence_list.append((id, self.sequence_list[i][1]))

            self.sequence_list = sequence_list
            self.num_tracks_per_seq = len(self.sequence_list) * [1]

        self.sequence_infos = {}

    def _preload_sequence_info(self, seq_id, anno_dict=None):
        objid, seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, 'Annotations', seq_name + '.json')

        with open(seq_path, 'r') as f:
            anno = json.load(f)

        if objid is None:
            info = {'bbox': anno['anno'], 'num_tracks': anno['num_tracks']}
        else:
            bbox = torch.tensor([b[objid] if objid in b else [-1, -1, -1, -1] for b in anno['anno']], dtype=torch.float)
            visible = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            info = {'bbox': bbox.tolist(), 'visible': visible}

        return info

    def is_mot_dataset(self):
        return self.multiobj

    def get_name(self):
        return 'ImagenetVIDMOT'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        if seq_id not in self.sequence_infos:
            self.sequence_infos[seq_id] = self._preload_sequence_info(seq_id)

        return self.sequence_infos[seq_id]

    def _get_frame(self, seq_path, frame_id):
        frame_path = self._get_frame_path(seq_path, frame_id)
        return self.image_loader(frame_path)

    def _get_sequence_path(self, seq_id):
        sequence = self.sequence_list[seq_id][1]
        return os.path.join(self.root, 'Data', sequence)

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06d}.JPEG'.format(frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            if isinstance(value, list) or torch.is_tensor(value):
                if isinstance(value[frame_ids[0]], dict):
                    anno_frames[key] = [{k: torch.tensor(v).clone() for k, v in value[f_id].items()} for f_id in
                                        frame_ids]
                else:
                    anno_frames[key] = [torch.tensor(value[f_id]).clone() for f_id in frame_ids]
            else:
                anno_frames[key] = value

        # added the class info to the meta info
        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

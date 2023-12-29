import os
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
import json
import torch
from collections import OrderedDict
from ltr.admin.environment import env_settings


class TAOBURST(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, multiobj=True):
        root = env_settings().tao_burst_dir if root is None else root
        super().__init__('TAOBURST', root, image_loader)

        self.multiobj = multiobj
        self.anno_path = os.path.join(root, 'TaoBurst.json')

        with open(self.anno_path, 'r') as f:
            self.annos = json.load(f)
            self.sequence_list = [(None, seq) for seq in self.annos.keys()]

        if multiobj == False:
            sequence_list = []
            for i in range(len(self.sequence_list)):
                info = self._preload_sequence_info(i)

                for id in info['bbox'][0].keys():
                    sequence_list.append((id, self.sequence_list[i][1]))

            self.sequence_list = sequence_list

        self.sequence_infos = {}

    def _preload_sequence_info(self, seq_id):
        objid, seq_name = self.sequence_list[seq_id]
        anno = self.annos[seq_name]

        if objid is None:
            info = {'bbox': anno['annotations'], 'num_tracks': len(anno['track_ids'])}
        else:
            bbox = torch.tensor([b[objid] if objid in b else [-1,-1,-1,-1] for b in anno['annotations']], dtype=torch.float)
            visible = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            info = {'bbox': bbox.tolist(), 'visible': visible}

        return info

    def is_mot_dataset(self):
        return self.multiobj

    def get_name(self):
        return 'TAOBURST'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        if seq_id not in self.sequence_infos:
            if self.annos is None:
                with open(self.anno_path, 'r') as f:
                    self.annos = json.load(f)

            self.sequence_infos[seq_id] = self._preload_sequence_info(seq_id)

        return self.sequence_infos[seq_id]

    def _get_frame(self, seq_path, seq_id, frame_id):
        frame_path = self._get_frame_path(seq_path, seq_id, frame_id)
        return self.image_loader(frame_path)

    def _get_sequence_path(self, seq_id):
        sequence = self.sequence_list[seq_id][1]
        return os.path.join(self.root, 'annotated_frames', self.annos[sequence]['split'], self.annos[sequence]['dataset_name'], self.annos[sequence]['seq_name'])

    def _get_frame_path(self, seq_path, seq_id, frame_id):
        return os.path.join(seq_path, self.annos[self.sequence_list[seq_id][1]]['annotated_image_paths'][frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None, frames_dict=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, seq_id, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            if isinstance(value, list) or torch.is_tensor(value):
                if isinstance(value[frame_ids[0]], dict):
                    anno_frames[key] = [{k: torch.tensor(v).clone() for k, v in value[f_id].items()} for f_id in frame_ids]
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


    def get_frame_annotation_period(self, seq_id):
        split = self.annos[self.sequence_list[seq_id][1]]['split']
        return 5 if split == 'train' else 30
